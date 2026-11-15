/**
 * One-shot Claude Code lifecycle: invoke the official Agent SDK, place its
 * real CLI process under the shared subprocess owner, map only strict SDK
 * success to completion, and dispose to whole-range quiescence.
 *
 * @module @deepseek-ai/dsh-subagent-claude-code/run
 */

import { randomUUID } from 'node:crypto'
import { accessSync, closeSync, constants, openSync, readSync, statSync } from 'node:fs'
import { extname, isAbsolute } from 'node:path'
import {
  query as officialQuery,
  type Options,
  type Query,
  type SDKMessage,
  type SDKResultMessage,
  type SpawnOptions,
} from '@anthropic-ai/claude-agent-sdk'
import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import { brandString } from '@deepseek-ai/dsh-brand'
import type { SessionId } from '@deepseek-ai/dsh-session'
import {
  settleRunResult,
  subprocessRunHandle,
  type SubagentResult,
  type SubagentRun,
  type SubagentStartRequest,
  type SubagentStopReason,
} from '@deepseek-ai/dsh-subagent'
import {
  scrubbedParentEnv,
  type SubprocessHandle,
  type SubprocessOutcome,
  type SubprocessSpawnSpec,
} from '@deepseek-ai/dsh-subprocess'
import {
  claudeSpawnSpec,
  ManagedClaudeCodeProcess,
} from './process.ts'

/** Default POSIX grace between subprocess termination tiers. */
export const DEFAULT_DISPOSE_GRACE_MS = 3_000

/** Claude Code permission modes that cannot wait for a human response. */
export const CLAUDE_CODE_PERMISSION_MODES = [
  'dontAsk',
  'acceptEdits',
  'auto',
  'plan',
  'bypassPermissions',
] as const satisfies readonly NonNullable<Options['permissionMode']>[]

/** Profile-selectable non-interactive Claude Code permission mode. */
export type ClaudeCodePermissionMode = typeof CLAUDE_CODE_PERMISSION_MODES[number]

/** Safe default for unattended Claude Code runs. */
export const DEFAULT_CLAUDE_CODE_PERMISSION_MODE: ClaudeCodePermissionMode = 'dontAsk'

const SUPPORTED_UNATTENDED_DIALOG_KINDS = [
  'refusal_fallback_prompt',
] satisfies NonNullable<Options['supportedDialogKinds']>

type ClaudeCodeFailureStage =
  | 'query-start'
  | 'query-run'
  | 'process'
  | 'teardown'

type ClaudeCodeFailureCategory =
  | 'limit'
  | 'product-error'
  | 'invalid-result'
  | 'process'
  | 'unknown'

interface ClaudeCodeFailureFacts {
  readonly stage: ClaudeCodeFailureStage
  readonly category: ClaudeCodeFailureCategory
  readonly outcome?: SubprocessOutcome | undefined
}

function failureDiagnostic(facts: ClaudeCodeFailureFacts): string {
  const fields = [
    'product: Claude Code',
    `stage: ${facts.stage}`,
    `category: ${facts.category}`,
  ]
  const exitCode = facts.outcome?.exitCode
  if (exitCode !== null && exitCode !== undefined) {
    fields.push(`exit code: ${exitCode}`)
  }
  const signal = facts.outcome?.signal
  if (signal !== null && signal !== undefined) {
    fields.push(`signal: ${signal}`)
  }
  return `Product subagent failure (${fields.join('; ')})`
}

class ClaudeCodeFailure extends Error {
  constructor(
    readonly facts: ClaudeCodeFailureFacts,
    cause?: unknown,
  ) {
    super(
      `subagent-claude-code: ${failureDiagnostic(facts)}`,
      cause === undefined ? undefined : { cause },
    )
    this.name = 'ClaudeCodeFailure'
  }
}

function sdkFailureCategory(
  subtype: string,
): ClaudeCodeFailureCategory {
  switch (subtype) {
    case 'error_max_turns':
    case 'error_max_budget_usd':
    case 'error_max_structured_output_retries':
      return 'limit'
    case 'error_during_execution':
      return 'product-error'
    default:
      return 'unknown'
  }
}

/**
 * Hide an unpublished product startup failure behind fixed safe facts.
 * @param cause - original host-side failure retained only on the Error cause chain.
 * @returns a rejection safe to expose through the subagent start boundary.
 */
export function claudeCodeStartupFailure(cause: unknown): Error {
  return new ClaudeCodeFailure({
    stage: 'query-start',
    category: 'unknown',
  }, cause)
}

function unattendedDiagnostic(
  mode: ClaudeCodePermissionMode,
  request: 'tool permission' | 'MCP elicitation' | 'user dialog',
  decision: 'denied' | 'declined' | 'cancelled',
  reason: string,
): string {
  return `Claude Code unattended decision (mode: ${mode}; request: ${request}; decision: ${decision}): ${reason}`
}

/* jscpd:ignore-start -- sibling providers intentionally keep product-private
 * run inputs and error normalization instead of adding a shared lifecycle owner. */
/** Fully resolved inputs for one official Claude Agent SDK query. */
export interface ClaudeCodeRunSpec {
  /** Selected child working directory supplied to the SDK and real CLI. */
  readonly cwd: string
  /** Profile-selected native model; omitted to preserve Claude settings. */
  readonly model?: string
  /** Explicit absolute native executable; omitted to retain the SDK's pinned runtime. */
  readonly pathToClaudeCodeExecutable?: string
  /** Profile-selected native non-interactive permission mode. */
  readonly permissionMode: ClaudeCodePermissionMode
  /** Explicit deployment/test environment layered after shared scrubbing. */
  readonly env: Record<string, string>
  /** Subprocess termination grace passed to the shared managed-range owner. */
  readonly disposeGraceMs: number
  /** Shared subprocess service spawn operation. */
  readonly spawn: (spec: SubprocessSpawnSpec) => SubprocessHandle
  /** Host diagnostic sink for a product failure kept outside model-visible text. */
  readonly onError?: (error: Error, stopReason: SubagentStopReason) => void
}

function thrown(value: unknown): Error {
  /* v8 ignore next -- typed SDK and subprocess failures reject with Error. */
  return value instanceof Error ? value : new Error(String(value))
}

/** Read live request cancellation across awaited startup cleanup. */
function isAborted(signal: AbortSignal): boolean {
  return signal.aborted
}

/* jscpd:ignore-end */

/**
 * Validate and preserve the one-shot task before crossing the SDK boundary.
 * @param prompt - task content accepted from the shared subagent service.
 * @returns the exact text sequence as one SDK prompt.
 */
export function textTask(prompt: readonly ContentBlock[]): string {
  if (prompt.length === 0) {
    throw new Error('subagent-claude-code: the one-shot task must contain only text blocks')
  }
  const texts: string[] = []
  for (const block of prompt) {
    if (block.type !== 'text') {
      throw new Error('subagent-claude-code: the one-shot task must contain only text blocks')
    }
    texts.push(block.text)
  }
  if (texts.every(text => text.trim().length === 0)) {
    throw new Error('subagent-claude-code: the one-shot task must not be empty')
  }
  return texts.join('')
}

/**
 * Strictly derive the only SDK result that can complete a shared run.
 * @param message - an official discriminated result union.
 * @returns exact final text for a successful, non-error result.
 */
export function successfulResult(message: SDKResultMessage): string {
  if (message.subtype !== 'success') {
    const category = sdkFailureCategory(message.subtype)
    const detail = category === 'unknown'
      ? undefined
      : message.errors.join('; ')
    throw new ClaudeCodeFailure(
      { stage: 'query-run', category },
      detail === undefined || detail.length === 0
        ? undefined
        : new Error(detail),
    )
  }
  if (message.is_error || message.result.trim().length === 0) {
    throw new ClaudeCodeFailure({
      stage: 'query-run',
      category: 'invalid-result',
    })
  }
  return message.result
}

/**
 * Consume the complete SDK stream and require one strict success plus normal
 * iterator completion.
 * @param query - published official SDK query.
 * @param onPermissionDenied - records a safe fact when the SDK reports native denial.
 * @param onResult - records that the SDK supplied a terminal result message.
 * @returns the completed shared result.
 */
export async function consumeClaudeQuery(
  query: AsyncIterable<SDKMessage>,
  onPermissionDenied?: () => void,
  onResult?: () => void,
): Promise<SubagentResult> {
  let answer: string | undefined
  for await (const message of query) {
    if (message.type === 'system' && message.subtype === 'permission_denied') {
      onPermissionDenied?.()
      continue
    }
    if (message.type !== 'result') continue
    onResult?.()
    answer = successfulResult(message)
  }
  if (answer === undefined) {
    throw new ClaudeCodeFailure({
      stage: 'query-run',
      category: 'invalid-result',
    })
  }
  return {
    output: [{ type: 'text', text: answer }],
    stopReason: 'completed',
  }
}

/**
 * Close the official query, terminate the managed range, and wait for the
 * subprocess owner to prove it is quiescent.
 * @param query - official SDK query, when creation reached that point.
 * @param child - shared-service handle that owns the CLI managed range, including
 * a published handle whose direct result later rejects.
 */
export async function disposeClaudeCodeChild(
  query: Pick<Query, 'close'> | undefined,
  child: SubprocessHandle,
): Promise<void> {
  const failures: Error[] = []
  let outcome: SubprocessOutcome | undefined
  void child.done.then(
    (value) => { outcome = value },
    () => {},
  )
  try {
    query?.close()
  } catch (error: unknown) {
    failures.push(thrown(error))
  }

  child.terminate()
  try {
    await child.waitForExit()
  } catch (error: unknown) {
    failures.push(thrown(error))
  }

  const firstFailure = failures[0]
  if (firstFailure !== undefined) {
    const facts = {
      stage: 'teardown',
      category: 'unknown',
      outcome,
    } as const
    const cause = failures.length === 1
      ? firstFailure
      : new AggregateError(failures, 'Claude Code teardown failures')
    throw new ClaudeCodeFailure(facts, cause)
  }
  await child.done.catch(() => {})
}

/** Recognize a bounded, structured host executable header, without proving loader compatibility. */
function hasNativeExecutableHeader(descriptor: number, size: number): boolean {
  const readHeader = (position: number, length: number): Buffer => {
    const bytes = Buffer.alloc(length)
    return bytes.subarray(0, readSync(descriptor, bytes, 0, length, position))
  }
  const header = readHeader(0, 64)
  if (header.length < 28) return false
  const magic = header.subarray(0, 4).toString('hex')
  if (process.platform === 'win32') {
    if (header.length !== 64 || header.readUInt16LE(0) !== 0x5a4d) return false
    const offset = header.readUInt32LE(60)
    if (offset < 64 || offset > size - 26) return false
    const pe = readHeader(offset, 26)
    if (pe.length !== 26 || pe.readUInt32LE(0) !== 0x00004550) return false
    const sections = pe.readUInt16LE(6)
    const optionalSize = pe.readUInt16LE(20)
    const characteristics = pe.readUInt16LE(22)
    const optionalMagic = pe.readUInt16LE(24)
    const minimumOptionalSize = optionalMagic === 0x10b ? 96 : optionalMagic === 0x20b ? 112 : 0
    return pe.readUInt16LE(4) !== 0 && sections > 0
      && (characteristics & 0x0002) !== 0 && (characteristics & 0x2000) === 0
      && minimumOptionalSize > 0 && optionalSize >= minimumOptionalSize
      && offset + 24 + optionalSize + sections * 40 <= size
  }
  if (process.platform === 'linux') {
    if (magic !== '7f454c46' || ![1, 2].includes(header.readUInt8(4))
      || ![1, 2].includes(header.readUInt8(5)) || header[6] !== 1
      || !header.subarray(9, 16).every(byte => byte === 0)) return false
    const little = header[5] === 1
    const u16 = (offset: number): number => little ? header.readUInt16LE(offset) : header.readUInt16BE(offset)
    const u32 = (offset: number): number => little ? header.readUInt32LE(offset) : header.readUInt32BE(offset)
    const is64 = header[4] === 2
    const headerSize = is64 ? 64 : 52
    if (header.length < headerSize || ![2, 3].includes(u16(16)) || u16(18) === 0 || u32(20) !== 1) return false
    const offset = is64
      ? (little ? header.readBigUInt64LE(32) : header.readBigUInt64BE(32))
      : BigInt(u32(28))
    const count = u16(is64 ? 56 : 44)
    const entrySize = is64 ? 56 : 32
    if (u16(is64 ? 52 : 40) !== headerSize || u16(is64 ? 54 : 42) !== entrySize
      || count === 0 || offset < BigInt(headerSize)
      || offset + BigInt(count * entrySize) > BigInt(size)) return false
    // The format's uint16 count bounds this table below 4 MiB.
    const programs = readHeader(Number(offset), count * entrySize)
    if (programs.length !== count * entrySize) return false
    const p32 = (position: number): number => little ? programs.readUInt32LE(position) : programs.readUInt32BE(position)
    const p64 = (position: number): bigint => little ? programs.readBigUInt64LE(position) : programs.readBigUInt64BE(position)
    let executableLoad = false
    for (let index = 0; index < count; index++) {
      const start = index * entrySize
      if (p32(start) !== 1) continue // PT_LOAD
      const fileOffset = is64 ? p64(start + 8) : BigInt(p32(start + 4))
      const fileSize = is64 ? p64(start + 32) : BigInt(p32(start + 16))
      const memorySize = is64 ? p64(start + 40) : BigInt(p32(start + 20))
      if (fileSize > memorySize || fileOffset + fileSize > BigInt(size)) return false
      if (fileSize > 0n && (p32(start + (is64 ? 4 : 24)) & 1) !== 0) executableLoad = true
    }
    return executableLoad
  }
  if (process.platform !== 'darwin') return false
  const isMachO = (bytes: Buffer, sliceSize: number): boolean => {
    if (bytes.length < 28) return false
    const signature = bytes.subarray(0, 4).toString('hex')
    if (!['feedface', 'cefaedfe', 'feedfacf', 'cffaedfe'].includes(signature)) return false
    const little = signature === 'cefaedfe' || signature === 'cffaedfe'
    const u32 = (offset: number): number => little ? bytes.readUInt32LE(offset) : bytes.readUInt32BE(offset)
    const headerSize = signature === 'feedfacf' || signature === 'cffaedfe' ? 32 : 28
    if (bytes.length < headerSize || u32(4) === 0 || u32(12) !== 2) return false
    const commands = u32(16)
    const commandBytes = u32(20)
    return commands > 0 && commandBytes >= commands * 8 && headerSize + commandBytes <= sliceSize
      && (headerSize !== 32 || u32(28) === 0)
  }
  if (!['cafebabe', 'bebafeca', 'cafebabf', 'bfbafeca'].includes(magic)) return isMachO(header, size)
  const little = magic === 'bebafeca' || magic === 'bfbafeca'
  const u32 = (offset: number): number => little ? header.readUInt32LE(offset) : header.readUInt32BE(offset)
  const u64 = (offset: number): bigint => little ? header.readBigUInt64LE(offset) : header.readBigUInt64BE(offset)
  const is64 = magic === 'cafebabf' || magic === 'bfbafeca'
  const count = u32(4)
  const tableSize = 8 + count * (is64 ? 32 : 20)
  if (count === 0 || tableSize > size || header.length < (is64 ? 40 : 28)) return false
  const offset = is64 ? u64(16) : BigInt(u32(16))
  const sliceSize = is64 ? u64(24) : BigInt(u32(20))
  if (offset < BigInt(tableSize) || sliceSize < 28n || offset + sliceSize > BigInt(size)) return false
  return isMachO(readHeader(Number(offset), Number(sliceSize < 32n ? sliceSize : 32n)), Number(sliceSize))
}

/**
 * Build the fixed official SDK options for one one-shot provider run.
 * @param spec - Workspace, environment, process service, and disposal policy.
 * @param controller - per-run cancellation owner.
 * @param capture - receives the shared child and SDK-facing process synchronously.
 * @param captureDiagnostic - receives safe facts from unattended interaction callbacks.
 * @returns options that inherit native settings while disabling persistence and user questions.
 */
export function claudeQueryOptions(
  spec: ClaudeCodeRunSpec,
  controller: AbortController,
  capture: (
    child: SubprocessHandle,
    process: ManagedClaudeCodeProcess,
  ) => void,
  captureDiagnostic: (diagnostic: string) => void,
): Options {
  const executable = spec.pathToClaudeCodeExecutable
  if (executable !== undefined) {
    if (!isAbsolute(executable)
      || /\.(?:cmd|bat|ps1|sh|bash|zsh|fish|js|mjs|cjs)$/i.test(executable)
      || (process.platform === 'win32' && extname(executable).toLowerCase() !== '.exe')) {
      throw new Error('subagent-claude-code: pathToClaudeCodeExecutable must be an absolute native executable path')
    }
    const executableStat = statSync(executable)
    if (!executableStat.isFile()) {
      throw new Error('subagent-claude-code: pathToClaudeCodeExecutable must reference a regular file')
    }
    accessSync(executable, constants.X_OK)
    const descriptor = openSync(executable, 'r')
    try {
      // Inspect the target file, including through symlinks; names alone do not
      // distinguish native executables from extensionless script wrappers.
      if (!hasNativeExecutableHeader(descriptor, executableStat.size)) {
        throw new Error('subagent-claude-code: pathToClaudeCodeExecutable must contain a native executable header, not a script')
      }
    } finally { closeSync(descriptor) }
  }
  return {
    abortController: controller,
    cwd: spec.cwd,
    ...spec.model === undefined ? {} : { model: spec.model },
    ...executable === undefined ? {} : { pathToClaudeCodeExecutable: executable },
    env: { ...scrubbedParentEnv(), ...spec.env },
    persistSession: false,
    disallowedTools: spec.permissionMode === 'plan'
      ? ['AskUserQuestion', 'ExitPlanMode']
      : ['AskUserQuestion'],
    permissionMode: spec.permissionMode,
    ...spec.permissionMode === 'bypassPermissions'
      ? { allowDangerouslySkipPermissions: true }
      : {
        canUseTool: () => {
          captureDiagnostic(unattendedDiagnostic(
            spec.permissionMode,
            'tool permission',
            'denied',
            'the provider does not request human approval',
          ))
          return Promise.resolve({
            behavior: 'deny' as const,
            message: 'This unattended Claude Code subagent cannot request human approval.',
          })
        },
      },
    onElicitation: () => {
      captureDiagnostic(unattendedDiagnostic(
        spec.permissionMode,
        'MCP elicitation',
        'declined',
        'the provider does not collect interactive MCP input',
      ))
      return Promise.resolve({ action: 'decline' })
    },
    onUserDialog: () => {
      captureDiagnostic(unattendedDiagnostic(
        spec.permissionMode,
        'user dialog',
        'cancelled',
        'the provider does not render blocking dialogs',
      ))
      return Promise.resolve({ behavior: 'cancelled' as const })
    },
    supportedDialogKinds: SUPPORTED_UNATTENDED_DIALOG_KINDS,
    spawnClaudeCodeProcess: (options: SpawnOptions) => {
      const child = spec.spawn({
        ...claudeSpawnSpec(options, spec.disposeGraceMs),
        ...executable === undefined ? {} : { allowShellFallback: false },
      })
      const process = new ManagedClaudeCodeProcess(child)
      capture(child, process)
      return process
    },
  }
}

/**
 * Start one official Claude Agent SDK query and publish its one-shot run.
 * @param request - resolved shared subagent request.
 * @param spec - Workspace, environment, process service, and diagnostic policy.
 * @returns the published run after both Query and the real CLI handle exist.
 */
export async function startClaudeCodeRun(
  request: SubagentStartRequest,
  spec: ClaudeCodeRunSpec,
): Promise<SubagentRun> {
  const prompt = textTask(request.prompt)
  if (request.signal.aborted) {
    throw new Error('subagent-claude-code: request was aborted before SDK startup')
  }

  const controller = new AbortController()
  const requestCancel = (): void => {
    if (!controller.signal.aborted) {
      controller.abort(new Error('subagent-claude-code: run cancelled locally'))
    }
  }
  const onAbort = (): void => { requestCancel() }
  request.signal.addEventListener('abort', onAbort, { once: true })
  const reportFailure = (error: Error): void => {
    try {
      spec.onError?.(error, 'error')
    } catch {
      // Host diagnostic logging cannot replace the product failure.
    }
  }

  let child: SubprocessHandle | undefined
  let childFailure: Error | undefined
  let childProcessFailure: Promise<never> | undefined
  let query: Query | undefined
  let managedProcess: ManagedClaudeCodeProcess | undefined
  let diagnostic: string | undefined
  const capturePermissionDiagnostic = (value: string): void => {
    diagnostic = value
  }
  const prependFailureDiagnostic = (facts: ClaudeCodeFailureFacts): void => {
    const failure = failureDiagnostic(facts)
    diagnostic = diagnostic === undefined
      ? failure
      : `${failure}\n${diagnostic}`
  }
  const captureChild = (
    captured: SubprocessHandle,
    process: ManagedClaudeCodeProcess,
  ): void => {
    child = captured
    managedProcess = process
    childProcessFailure = captured.done.then(
      () => new Promise<never>(() => {}),
      (error: unknown) => {
        childFailure = thrown(error)
        throw childFailure
      },
    )
    void childProcessFailure.catch(() => {})
  }
  try {
    query = officialQuery({
      prompt,
      options: claudeQueryOptions(
        spec,
        controller,
        captureChild,
        capturePermissionDiagnostic,
      ),
    })
    if (child === undefined || childProcessFailure === undefined) {
      throw new Error(
        'subagent-claude-code: official SDK did not publish a controllable Claude Code process',
      )
    }
    if (isAborted(controller.signal)) {
      throw new Error('subagent-claude-code: request was aborted before SDK startup')
    }
  } catch (error: unknown) {
    request.signal.removeEventListener('abort', onAbort)
    const cancelledBeforeCleanup = controller.signal.aborted
    // Let child.done publish a concurrently observed exit before classification.
    await Promise.resolve()
    const startupOutcome = managedProcess?.outcome
    const startupFacts = {
      stage: 'query-start',
      category: 'unknown',
      outcome: startupOutcome,
    } as const
    const startupFailure = (cause: unknown = childFailure ?? error): ClaudeCodeFailure => new ClaudeCodeFailure(
      startupFacts,
      thrown(cause),
    )
    requestCancel()
    if (child !== undefined) {
      try {
        await disposeClaudeCodeChild(query, child)
      } catch (disposeError: unknown) {
        const failure = startupFailure()
        const cleanupFailure = thrown(disposeError)
        const aggregate = new AggregateError(
          [failure, cleanupFailure],
          `${failure.message}; ${cleanupFailure.message}`,
        )
        reportFailure(aggregate)
        throw aggregate
      }
      if (cancelledBeforeCleanup || isAborted(request.signal)) {
        throw new Error('subagent-claude-code: request was aborted before SDK startup')
      }
      const failure = startupFailure()
      reportFailure(failure)
      throw failure
    } else if (query !== undefined) {
      try {
        query.close()
      } catch (disposeError: unknown) {
        const failure = startupFailure()
        const cleanupFailure = new ClaudeCodeFailure({
          stage: 'teardown',
          category: 'unknown',
        }, thrown(disposeError))
        const aggregate = new AggregateError(
          [failure, cleanupFailure],
          `${failure.message}; ${cleanupFailure.message}`,
        )
        reportFailure(aggregate)
        throw aggregate
      }
    }
    if (cancelledBeforeCleanup || isAborted(request.signal)) {
      throw new Error('subagent-claude-code: request was aborted before SDK startup')
    }
    const failure = startupFailure()
    reportFailure(failure)
    throw failure
  }

  const publishedQuery = query
  const publishedChild = child
  const publishedProcessFailure = childProcessFailure
  let receivedResult = false
  const result = settleRunResult({
    attempt: async () => {
      try {
        return await Promise.race([
          consumeClaudeQuery(publishedQuery, () => {
            capturePermissionDiagnostic(unattendedDiagnostic(
              spec.permissionMode,
              'tool permission',
              'denied',
              'Claude Code denied the request before an interactive prompt',
            ))
          }, () => {
            receivedResult = true
          }),
          publishedProcessFailure,
        ])
      } catch (error: unknown) {
        const processOutcome = managedProcess?.outcome
        let facts: ClaudeCodeFailureFacts
        if (error instanceof ClaudeCodeFailure) {
          facts = { ...error.facts, outcome: processOutcome }
        } else if (processOutcome !== undefined && !receivedResult) {
          facts = {
            stage: 'process',
            category: 'process',
            outcome: processOutcome,
          }
        } else {
          facts = {
            stage: 'query-run',
            category: 'unknown',
            outcome: processOutcome,
          }
        }
        prependFailureDiagnostic(facts)
        // Keep the SDK category and cause; the diagnostic adds later process facts.
        throw error instanceof ClaudeCodeFailure
          ? error
          : new ClaudeCodeFailure(facts, thrown(error))
      }
    },
    collectOutput: () => [],
    collectDiagnostic: () => diagnostic,
    cancelled: () => controller.signal.aborted,
    onError: spec.onError,
    signal: request.signal,
    onAbort,
  })

  return subprocessRunHandle({
    id: brandString<SessionId>(randomUUID()),
    result,
    signal: request.signal,
    onAbort,
    requestCancel,
    teardown: async () => {
      try {
        await disposeClaudeCodeChild(publishedQuery, publishedChild)
      } catch (error: unknown) {
        const failure = thrown(error)
        reportFailure(failure)
        throw failure
      }
    },
  })
}
