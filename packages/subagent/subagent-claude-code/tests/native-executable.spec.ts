import * as fs from 'node:fs'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { resolve } from 'node:path'
import { describe, expect, it, vi } from 'vitest'
import { claudeQueryOptions } from '../src/run.ts'

vi.mock('node:fs', async importOriginal => ({ ...await importOriginal<typeof import('node:fs')>() }))

type Host = 'win32' | 'linux' | 'darwin'

/** Make a structurally plausible header; these fixtures are never executed. */
function peHeader(is64 = true): Buffer {
  const bytes = Buffer.alloc(512)
  bytes.write('MZ')
  bytes.writeUInt32LE(64, 60)
  bytes.write('PE\0\0', 64)
  bytes.writeUInt16LE(0x8664, 68)
  bytes.writeUInt16LE(1, 70)
  bytes.writeUInt16LE(is64 ? 112 : 96, 84)
  bytes.writeUInt16LE(2, 86)
  bytes.writeUInt16LE(is64 ? 0x20b : 0x10b, 88)
  return bytes
}

function elfHeader(is64 = true, little = true): Buffer {
  const bytes = Buffer.alloc(512)
  bytes.set([0x7f, 0x45, 0x4c, 0x46, is64 ? 2 : 1, little ? 1 : 2, 1])
  const u16 = (value: number, offset: number): void => {
    if (little) bytes.writeUInt16LE(value, offset)
    else bytes.writeUInt16BE(value, offset)
  }
  const u32 = (value: number, offset: number): void => {
    if (little) bytes.writeUInt32LE(value, offset)
    else bytes.writeUInt32BE(value, offset)
  }
  const u64 = (value: bigint, offset: number): void => {
    if (little) bytes.writeBigUInt64LE(value, offset)
    else bytes.writeBigUInt64BE(value, offset)
  }
  u16(2, 16)
  u16(62, 18)
  u32(1, 20)
  const headerSize = is64 ? 64 : 52
  if (is64) u64(64n, 32)
  else u32(52, 28)
  u16(headerSize, is64 ? 52 : 40)
  u16(is64 ? 56 : 32, is64 ? 54 : 42)
  u16(2, is64 ? 56 : 44)
  u32(6, headerSize) // A non-load program header before the executable load.
  const load = headerSize + (is64 ? 56 : 32)
  u32(1, load)
  u32(5, load + (is64 ? 4 : 24))
  if (is64) {
    u64(512n, load + 32)
    u64(512n, load + 40)
  } else {
    u32(512, load + 16)
    u32(512, load + 20)
  }
  return bytes
}

function machHeader(is64 = true, little = true): Buffer {
  const bytes = Buffer.alloc(512)
  const u32 = (value: number, offset: number): void => {
    if (little) bytes.writeUInt32LE(value, offset)
    else bytes.writeUInt32BE(value, offset)
  }
  u32(is64 ? 0xfeedfacf : 0xfeedface, 0)
  u32(0x01000007, 4)
  u32(2, 12)
  u32(1, 16)
  u32(8, 20)
  return bytes
}

function fatHeader(is64 = true, little = true): Buffer {
  const bytes = Buffer.alloc(1024)
  const u32 = (value: number, offset: number): void => {
    if (little) bytes.writeUInt32LE(value, offset)
    else bytes.writeUInt32BE(value, offset)
  }
  const u64 = (value: bigint, offset: number): void => {
    if (little) bytes.writeBigUInt64LE(value, offset)
    else bytes.writeBigUInt64BE(value, offset)
  }
  u32(is64 ? 0xcafebabf : 0xcafebabe, 0)
  u32(1, 4)
  if (is64) {
    u64(256n, 16)
    u64(512n, 24)
  } else {
    u32(256, 16)
    u32(512, 20)
  }
  bytes.set(machHeader(), 256)
  return bytes
}

/** Override only the reported host while invoking the synchronous preflight. */
function preflight(host: string, executable: string): void {
  const original = Object.getOwnPropertyDescriptor(process, 'platform')!
  Object.defineProperty(process, 'platform', { value: host, configurable: true })
  try {
    const spawn = vi.fn(() => { throw new Error('Preflight must never spawn') })
    const options = claudeQueryOptions({
      cwd: process.cwd(), permissionMode: 'dontAsk', env: {}, disposeGraceMs: 10,
      pathToClaudeCodeExecutable: executable, spawn,
    }, new AbortController(), () => {}, () => {})
    expect(options.pathToClaudeCodeExecutable).toBe(executable)
    expect(spawn).not.toHaveBeenCalled()
  } finally { Object.defineProperty(process, 'platform', original) }
}

function writeFixture(bytes: Buffer): { executable: string; dispose(): void } {
  const directory = mkdtempSync(resolve(tmpdir(), 'dsh-native-header-'))
  const executable = resolve(directory, 'worker.exe')
  writeFileSync(executable, bytes, { mode: 0o700 })
  return { executable, dispose: () => { rmSync(directory, { recursive: true, force: true }) } }
}

describe('native executable structured preflight', () => {
  const valid: [string, Host, Buffer][] = [
    ['PE32', 'win32', peHeader(false)], ['PE32+', 'win32', peHeader()],
    ...[false, true].flatMap(is64 => [false, true].flatMap(little => [
      [`ELF ${is64 ? 64 : 32}/${little ? 'LE' : 'BE'}`, 'linux', elfHeader(is64, little)],
      [`Mach-O ${is64 ? 64 : 32}/${little ? 'LE' : 'BE'}`, 'darwin', machHeader(is64, little)],
      [`Universal Mach-O ${is64 ? 64 : 32}/${little ? 'LE' : 'BE'}`, 'darwin', fatHeader(is64, little)],
    ] as [string, Host, Buffer][])),
  ]
  it.each(valid)('recognizes the %s structure without launching it', (_name, host, bytes) => {
    const fixture = writeFixture(bytes)
    try { preflight(host, fixture.executable) } finally { fixture.dispose() }
  })
  it('requires an .exe suffix for a Windows override on every host', () => {
    const fixture = writeFixture(peHeader())
    const extensionless = fixture.executable.slice(0, -4)
    try {
      fs.renameSync(fixture.executable, extensionless)
      expect(() => { preflight('win32', extensionless) }).toThrow('absolute native executable path')
    } finally { fixture.dispose() }
  })
  it.each(valid)('rejects the foreign %s format on other hosts', (_name, host, bytes) => {
    const fixture = writeFixture(bytes)
    try {
      for (const other of ['win32', 'linux', 'darwin', 'unsupported']) {
        if (other !== host) expect(() =>{  preflight(other, fixture.executable) }).toThrow('native executable header')
      }
    } finally { fixture.dispose() }
  })

  const invalid: [string, Host, Buffer][] = []
  const mutate = (name: string, host: Host, source: Buffer, change: (bytes: Buffer) => void): void => {
    change(source)
    invalid.push([name, host, source])
  }
  for (const host of ['win32', 'linux', 'darwin'] as const) {
    for (const magic of ['4d5a3d78', '7f454c46', 'cffaedfe', 'cafebabe']) {
      invalid.push([`${magic} shell prefix on ${host}`, host, Buffer.concat([Buffer.from(magic, 'hex'), Buffer.from('\nprintf wrapper_payload\n'.repeat(4))])])
    }
  }
  invalid.push(['truncated DOS', 'win32', peHeader().subarray(0, 63)])
  mutate('DOS offset overlaps header', 'win32', peHeader(), b => b.writeUInt32LE(4, 60))
  mutate('DOS offset outside file', 'win32', peHeader(), b => b.writeUInt32LE(500, 60))
  mutate('missing PE signature', 'win32', peHeader(), b => b.writeUInt32LE(0, 64))
  mutate('missing PE machine', 'win32', peHeader(), b => b.writeUInt16LE(0, 68))
  mutate('no PE sections', 'win32', peHeader(), b => b.writeUInt16LE(0, 70))
  mutate('non-executable PE', 'win32', peHeader(), b => b.writeUInt16LE(0, 86))
  mutate('PE DLL', 'win32', peHeader(), b => b.writeUInt16LE(0x2002, 86))
  mutate('unknown optional PE header', 'win32', peHeader(), b => b.writeUInt16LE(0, 88))
  mutate('short optional PE header', 'win32', peHeader(), b => b.writeUInt16LE(1, 84))
  mutate('PE section table outside file', 'win32', peHeader(), b => b.writeUInt16LE(100, 70))
  mutate('unknown ELF class', 'linux', elfHeader(), (b) => { b[4] = 3 })
  mutate('unknown ELF endianness', 'linux', elfHeader(), (b) => { b[5] = 3 })
  mutate('unknown ELF ident version', 'linux', elfHeader(), (b) => { b[6] = 2 })
  mutate('ELF text in required padding', 'linux', elfHeader(), (b) => { b[9] = 65 })
  mutate('non-executable ELF type', 'linux', elfHeader(), b => b.writeUInt16LE(1, 16))
  mutate('missing ELF machine', 'linux', elfHeader(), b => b.writeUInt16LE(0, 18))
  mutate('unknown ELF version', 'linux', elfHeader(), b => b.writeUInt32LE(2, 20))
  mutate('wrong ELF header size', 'linux', elfHeader(), b => b.writeUInt16LE(1, 52))
  mutate('wrong ELF program size', 'linux', elfHeader(), b => b.writeUInt16LE(1, 54))
  mutate('no ELF programs', 'linux', elfHeader(), b => b.writeUInt16LE(0, 56))
  mutate('ELF programs overlap header', 'linux', elfHeader(), b => b.writeBigUInt64LE(0n, 32))
  mutate('ELF programs outside file', 'linux', elfHeader(), b => b.writeBigUInt64LE(500n, 32))
  mutate('ELF empty load table with shell payload', 'linux', elfHeader(), (b) => { b.fill(0, 64, 176); b.write('\nprintf wrapper_payload\n', 180) })
  mutate('ELF load file data outside file', 'linux', elfHeader(), b => b.writeBigUInt64LE(512n, 128))
  mutate('ELF load memory shorter than file', 'linux', elfHeader(), b => b.writeBigUInt64LE(1n, 160))
  mutate('ELF non-executable load', 'linux', elfHeader(), b => b.writeUInt32LE(4, 124))
  mutate('ELF empty executable load', 'linux', elfHeader(), b => b.writeBigUInt64LE(0n, 152))
  invalid.push(['truncated ELF header', 'linux', elfHeader().subarray(0, 51)])
  mutate('missing Mach-O CPU', 'darwin', machHeader(), b => b.writeUInt32LE(0, 4))
  mutate('Mach-O library', 'darwin', machHeader(), b => b.writeUInt32LE(6, 12))
  mutate('no Mach-O commands', 'darwin', machHeader(), b => b.writeUInt32LE(0, 16))
  mutate('short Mach-O command table', 'darwin', machHeader(), b => b.writeUInt32LE(1, 20))
  mutate('Mach-O command table outside slice', 'darwin', machHeader(), b => b.writeUInt32LE(512, 20))
  mutate('nonzero Mach-O reserved word', 'darwin', machHeader(), b => b.writeUInt32LE(1, 28))
  invalid.push(['truncated Mach-O64 header', 'darwin', machHeader().subarray(0, 28)])
  mutate('no universal architectures', 'darwin', fatHeader(), b => b.writeUInt32LE(0, 4))
  mutate('universal architecture table outside file', 'darwin', fatHeader(), b => b.writeUInt32LE(100, 4))
  mutate('universal slice overlaps table', 'darwin', fatHeader(), b => b.writeBigUInt64LE(0n, 16))
  mutate('universal slice too short', 'darwin', fatHeader(), b => b.writeBigUInt64LE(1n, 24))
  mutate('universal truncated 64-bit slice', 'darwin', fatHeader(), b => b.writeBigUInt64LE(28n, 24))
  mutate('universal slice outside file', 'darwin', fatHeader(), b => b.writeBigUInt64LE(1024n, 24))
  mutate('universal slice is text', 'darwin', fatHeader(), b => b.fill(65, 256, 288))
  it.each([
    ['ELF program table', 'linux', elfHeader(), 64],
    ['universal Mach-O slice', 'darwin', fatHeader(), 276],
  ] as const)('rejects a %s truncated after metadata inspection', (_name, host, bytes, size) => {
    const fixture = writeFixture(bytes)
    const originalStat = fs.statSync
    const stat = vi.spyOn(fs, 'statSync').mockImplementationOnce((...args) => {
      const metadata = originalStat(...args)
      fs.truncateSync(fixture.executable, size)
      return metadata
    })
    try { expect(() =>{  preflight(host, fixture.executable) }).toThrow('native executable header') } finally {
      stat.mockRestore()
      fixture.dispose()
    }
  })
  it.each(invalid)('rejects %s before query or spawn', (_name, host, bytes) => {
    const fixture = writeFixture(bytes)
    try { expect(() =>{  preflight(host, fixture.executable) }).toThrow('native executable header') } finally { fixture.dispose() }
  })
})
