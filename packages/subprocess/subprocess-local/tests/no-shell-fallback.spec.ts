import { Buffer } from 'node:buffer'
import { ChildProcess, type SpawnOptions } from 'node:child_process'
import { chmodSync, existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { release, tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it, onTestFinished, vi } from 'vitest'
import type { SubprocessSpawnSpec } from '@deepseek-ai/dsh-subprocess'
import { spawnSubprocess, validateSubprocessSpec } from '../src/spawn.ts'
import { consumeLinuxLaunchRequest, linuxLaunchFilesFromLocator } from '../src/runner-protocol.ts'
import { SUBPROCESS_RUNNER_ENV } from '../src/runner-launch.ts'

function request(argv: string[], cwd = process.cwd()): SubprocessSpawnSpec {
  return {
    argv, cwd, graceMs: 200,
    stdio: { stdin: 'ignore', stdout: { maxBytes: 4096 }, stderr: { maxBytes: 4096 } },
  }
}

it.each([null, 0, 'false', {}])('rejects a malformed public shell-fallback policy %j', (value) => {
  const spec = request([process.execPath])
  Reflect.set(spec, 'allowShellFallback', value)
  expect(() => { validateSubprocessSpec(spec) }).toThrow('allowShellFallback must be a boolean')
})

it('fails closed on an unsupported no-shell platform before spawning', () => {
  const spawn = vi.fn()
  expect(() => spawnSubprocess({ ...request(['tool']), allowShellFallback: false }, { platform: 'aix', spawn }))
    .toThrow('cannot disable shell fallback on aix')
  expect(spawn).not.toHaveBeenCalled()
})

it('preserves synchronous default spawn failures', () => {
  expect(() => spawnSubprocess(request(['tool']), { spawn: () => { throw new Error('spawn failed') } }))
    .toThrow('spawn failed')
})

it('rejects an unconsumed bootstrap and retains its request until the child closes', async () => {
  const child = new ChildProcess()
  child.stdin = null
  child.stdout = null
  child.stderr = null
  onTestFinished(() => {
    child.emit('exit', 1, null)
    child.emit('close', 1, null)
  })
  let directory: string | undefined
  const handle = spawnSubprocess({ ...request(['tool']), allowShellFallback: false }, {
    platform: 'linux',
    spawn: (_program, _args, options) => {
      const locator = options.env?.[SUBPROCESS_RUNNER_ENV]
      if (typeof locator !== 'string') throw new Error('missing launch request')
      directory = linuxLaunchFilesFromLocator(locator).directory
      return child
    },
  })
  const done = expect(handle.done).rejects.toThrow('before consuming its launch request')
  child.emit('exit', 1, null)
  await done
  expect(existsSync(directory!)).toBe(true)
  child.emit('close', 1, null)
  expect(existsSync(directory!)).toBe(false)
  expect(await handle.waitForExit()).toBe(true)
})

it.each(['linux', 'darwin'] as const)('selects the private no-shell bootstrap on %s and cleans synchronous failures', (platform) => {
  let directory: string | undefined
  const spawn = vi.fn((_program: string, args: readonly string[], options: SpawnOptions): never => {
    const locator = options.env?.[SUBPROCESS_RUNNER_ENV]
    if (typeof locator !== 'string') throw new Error('missing private runner locator')
    directory = linuxLaunchFilesFromLocator(locator).directory
    expect(consumeLinuxLaunchRequest(locator)).toMatchObject({
      cwd: process.cwd(), allowShellFallback: false, pipeFds: [1, 2],
    })
    expect(args.slice(-3)).toEqual(['--', 'tool', 'literal argument'])
    expect(options.detached).toBe(true)
    throw new Error('synthetic spawn failure')
  })
  expect(() => spawnSubprocess({ ...request(['tool', 'literal argument']), allowShellFallback: false }, { platform, spawn }))
    .toThrow('synthetic spawn failure')
  expect(directory).toBeDefined()
  expect(existsSync(directory!)).toBe(false)
})

// A plausible ELF64 header and executable, in-bounds PT_LOAD cannot establish
// loadability: this nonzero machine number is unsupported by the host kernel.
function invalidElf(payload: string): Buffer {
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
  return Buffer.concat([header, Buffer.from(`\n${payload}`)])
}

describe.skipIf(process.platform !== 'linux' && process.platform !== 'darwin')('real POSIX no-shell execution', () => {
  it.each(['executable text', 'MZ script', 'loader-invalid ELF'])('refuses %s without executing the shell payload', async (format) => {
    const directory = mkdtempSync(join(tmpdir(), 'dsh-no-shell-'))
    const marker = join(directory, 'executed')
    const file = join(directory, 'native-tool')
    const payload = 'printf shell-fallback-ran > executed\n'
    writeFileSync(file, format === 'loader-invalid ELF' ? invalidElf(payload) : `${format === 'MZ script' ? 'MZ' : 'TEXT'}=x\n${payload}`)
    chmodSync(file, 0o700)
    const handle = spawnSubprocess({ ...request([file], directory), allowShellFallback: false })
    try {
      if (format === 'MZ script') {
        const outcome = await handle.done.catch((error: unknown) => {
          expect(error).toMatchObject({ code: 'ENOEXEC' })
          return undefined
        })
        if (outcome !== undefined) {
          // WSL may dispatch MZ to its Windows loader before ENOEXEC.
          expect(process.platform).toBe('linux')
          expect(release().toLowerCase()).toContain('microsoft')
          expect(outcome).toEqual({ exitCode: 1, signal: null })
          expect(handle.collected.stderr?.readFrom(0).text).toBe(`${file}: Invalid argument\n`)
          expect(handle.collected.stdout?.readFrom(0).text).toBe('')
        }
      } else {
        await expect(handle.done).rejects.toMatchObject({ code: 'ENOEXEC' })
      }
      await handle.waitForExit()
      expect(existsSync(marker)).toBe(false)
    } finally {
      handle.terminate()
      await handle.waitForExit()
      rmSync(directory, { recursive: true, force: true })
    }
  }, 20_000)

  it.each([undefined, true])('preserves executable-text fallback with policy %s', async (allowShellFallback) => {
    const directory = mkdtempSync(join(tmpdir(), 'dsh-shell-default-'))
    const file = join(directory, 'native-tool')
    writeFileSync(file, 'TEXT=x\nprintf shell-fallback-ran\n', { mode: 0o700 })
    const handle = spawnSubprocess({ ...request([file], directory), ...allowShellFallback === undefined ? {} : { allowShellFallback } })
    try {
      await expect(handle.done).resolves.toEqual({ exitCode: 0, signal: null })
      await handle.waitForExit()
      expect(handle.collected.stdout?.readFrom(0).text).toBe('shell-fallback-ran')
    } finally {
      handle.terminate()
      await handle.waitForExit()
      rmSync(directory, { recursive: true, force: true })
    }
  })

  it('runs a real executable with literal argv, target environment and cwd', async () => {
    const directory = mkdtempSync(join(tmpdir(), 'dsh-native-valid-'))
    const handle = spawnSubprocess({
      ...request([process.execPath, '-e', 'process.stdout.write(JSON.stringify([process.argv[1],process.cwd(),process.env.TEST_LITERAL]))', 'a;$b'], directory),
      allowShellFallback: false, env: { TEST_LITERAL: 'literal-value' },
    })
    try {
      await expect(handle.done).resolves.toEqual({ exitCode: 0, signal: null })
      await handle.waitForExit()
      expect(JSON.parse(handle.collected.stdout!.readFrom(0).text)).toEqual(['a;$b', directory, 'literal-value'])
    } finally {
      handle.terminate()
      await handle.waitForExit()
      rmSync(directory, { recursive: true, force: true })
    }
  }, 20_000)
})
