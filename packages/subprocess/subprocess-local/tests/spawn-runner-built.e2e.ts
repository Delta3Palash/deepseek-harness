import { spawn, spawnSync } from 'node:child_process'
import { Buffer } from 'node:buffer'
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import type { SubprocessSpawnSpec } from '@deepseek-ai/dsh-subprocess'
import {
  cleanupLinuxLaunchFiles,
  createLinuxLaunchFiles,
  readLinuxStartupError,
} from '../src/runner-protocol.ts'
import {
  runnerEnvironment,
  runnerInvocationAvailable,
  SUBPROCESS_RUNNER_ENV,
  targetEnvironment,
} from '../src/runner-launch.ts'
import type { RunnerInvocation } from '../src/runner-launch.ts'
import { bindManagedProcess } from '../src/spawn.ts'
import { launchWindowsJob } from '../src/windows-job.ts'

const repoRoot = resolve(import.meta.dirname, '../../../..')
const sourceRunner = resolve(repoRoot, 'packages/subprocess/subprocess-local/src/bin.ts')
const builtRunner = fileURLToPath(import.meta.resolve('@deepseek-ai/dsh-subprocess-local/runner'))

function targetEnv(): Record<string, string> {
  return {
    ...Object.fromEntries(Object.entries(process.env).filter((entry): entry is [string, string] => entry[1] !== undefined)),
    [SUBPROCESS_RUNNER_ENV]: 'target-collision-restored',
  }
}

async function executePosix(invocation: RunnerInvocation): Promise<{ status: number | null; stdout: string; stderr: string }> {
  const files = createLinuxLaunchFiles({ cwd: repoRoot, env: targetEnv() })
  try {
    const child = spawn(invocation[0], [
      ...invocation.slice(1),
      '--',
      process.execPath,
      '--input-type=module',
      '--eval',
      `process.stdout.write(process.argv[0]+'|'+process.cwd()+'|'+process.env.${SUBPROCESS_RUNNER_ENV})`,
    ], {
      env: runnerEnvironment(files.requestPath, invocation),
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let stdout = ''
    let stderr = ''
    child.stdout.on('data', (chunk: Buffer) => { stdout += chunk.toString() })
    child.stderr.on('data', (chunk: Buffer) => { stderr += chunk.toString() })
    const status = await new Promise<number | null>((resolveExit, rejectExit) => {
      child.once('error', rejectExit)
      child.once('exit', resolveExit)
    })
    return { status, stdout, stderr }
  } finally {
    cleanupLinuxLaunchFiles(files)
  }
}

async function executeWindows(invocation: RunnerInvocation): Promise<{ status: number | null; stdout: string; stderr: string }> {
  const request: SubprocessSpawnSpec = {
    argv: [
      process.execPath,
      '--input-type=module',
      '--eval',
      `process.stdout.write(process.argv[0]+'|'+process.cwd()+'|'+process.env.${SUBPROCESS_RUNNER_ENV})`,
    ],
    cwd: repoRoot,
    env: targetEnv(),
    stdio: {
      stdin: 'ignore',
      stdout: { maxBytes: 64_000 },
      stderr: { maxBytes: 64_000 },
    },
    graceMs: 3_000,
  }
  const handle = bindManagedProcess(request, launchWindowsJob(
    request,
    targetEnvironment(request),
    { runnerInvocation: invocation },
  ))
  const outcome = await handle.done
  await handle.waitForExit()
  const stdout = handle.collected.stdout?.readFrom(0).text ?? ''
  const stderr = handle.collected.stderr?.readFrom(0).text ?? ''
  return { status: outcome.exitCode, stdout, stderr }
}

async function execute(invocation: RunnerInvocation): Promise<{ status: number | null; stdout: string; stderr: string }> {
  return process.platform === 'win32'
    ? executeWindows(invocation)
    : executePosix(invocation)
}

describe('subprocess-local runner artifacts', () => {
  it.skipIf(!existsSync(builtRunner) || !['linux', 'darwin'].includes(process.platform))(
    'rejects a loader-invalid structured ELF through the built bootstrap without shell execution', () => {
      const directory = mkdtempSync(join(tmpdir(), 'dsh-built-no-shell-'))
      const target = join(directory, 'forged-native')
      const header = Buffer.alloc(128)
      header.set([0x7f, 0x45, 0x4c, 0x46, 2, 1, 1])
      header.writeUInt16LE(2, 16)
      header.writeUInt16LE(0xffff, 18)
      header.writeUInt32LE(1, 20)
      header.writeBigUInt64LE(64n, 32)
      header.writeUInt16LE(64, 52)
      header.writeUInt16LE(56, 54)
      header.writeUInt16LE(1, 56)
      header.writeUInt32LE(1, 64)
      header.writeUInt32LE(5, 68)
      header.writeBigUInt64LE(128n, 96)
      header.writeBigUInt64LE(128n, 104)
      header.writeBigUInt64LE(4096n, 112)
      const files = createLinuxLaunchFiles({ cwd: directory, env: targetEnv(), allowShellFallback: false })
      try {
        writeFileSync(target, Buffer.concat([header, Buffer.from('\nprintf payload > executed\n')]), { mode: 0o700 })
        const invocation: RunnerInvocation = [process.execPath, builtRunner]
        const result = spawnSync(invocation[0], [...invocation.slice(1), '--', target], {
          cwd: directory, env: runnerEnvironment(files.requestPath, invocation), encoding: 'utf8', timeout: 10_000,
        })
        expect(result.error).toBeUndefined()
        expect(result.status, result.stderr).toBe(127)
        expect(readLinuxStartupError(files.startupErrorPath)).toMatchObject({ type: 'error', error: { code: 'ENOEXEC' } })
        expect(existsSync(join(directory, 'executed'))).toBe(false)
      } finally {
        cleanupLinuxLaunchFiles(files)
        rmSync(directory, { recursive: true, force: true })
      }
    }, 20_000,
  )

  it('executes the source entry through the provider-owned core', async () => {
    const result = await execute([process.execPath, '--import', 'tsx/esm', sourceRunner])
    expect(result).toEqual({
      status: 0,
      stdout: `${process.execPath}|${repoRoot}|target-collision-restored`,
      stderr: '',
    })
  }, 30_000)

  it.skipIf(!existsSync(builtRunner))('executes the built ./runner subpath through the same core', async () => {
    const invocation: RunnerInvocation = [process.execPath, builtRunner]
    expect(runnerInvocationAvailable(invocation)).toBe(true)
    const result = await execute(invocation)
    expect(result).toEqual({
      status: 0,
      stdout: `${process.execPath}|${repoRoot}|target-collision-restored`,
      stderr: '',
    })
  }, 30_000)
})
