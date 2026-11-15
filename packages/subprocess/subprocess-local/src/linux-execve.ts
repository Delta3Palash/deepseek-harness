/** Lazy libc execve and descriptor bindings used by the one-shot POSIX bootstrap. */

import { getSystemErrorMessage, getSystemErrorName } from 'node:util'
import { SUBPROCESS_CONTROL_FD } from '@deepseek-ai/dsh-subprocess/control'
import { createLazyRequire } from '@deepseek-ai/dsh-lazy-require'

const requireKoffi = createLazyRequire<typeof import('koffi')['default']>('koffi', import.meta.url)

/** Replace the current process image while preserving the supplied argv and environment. */
export type LinuxExecve = (
  file: string,
  argv: string[],
  env: Record<string, string>,
  control?: 'pipe',
  pipeFds?: readonly number[],
) => never

type NativeExecve = (
  file: string,
  argv: Array<string | null>,
  envp: Array<string | null>,
) => number

type NativeFcntl = (fd: number, command: number, argument: number) => number
type VariadicFcntl = (fd: number, command: number, type: 'int', argument: number) => number

const STANDARD_FILE_DESCRIPTORS = [0, 1, 2] as const
const F_GETFD = 1
const F_SETFD = 2
const FD_CLOEXEC = 1
const F_GETFL = 3
const F_SETFL = 4

let cachedExecve: LinuxExecve | undefined

function systemError(errno: number, syscall: string, path?: string): Error {
  const uvError = -errno
  const code = getSystemErrorName(uvError)
  const detail = getSystemErrorMessage(uvError)
  const subject = path === undefined ? syscall : `${syscall} '${path}'`
  const error = Object.assign(new Error(`${code}: ${detail}, ${subject}`), {
    code,
    errno: uvError,
    syscall,
  })
  return path === undefined ? error : Object.assign(error, { path })
}

/**
 * Load libc's execve and fcntl symbols on first use and retain the native bindings.
 * @returns a process-replacing execve operation that throws Node-style errors on failure.
 */
export function loadPosixExecve(): LinuxExecve {
  if (process.platform !== 'linux' && process.platform !== 'darwin') {
    throw new Error('direct POSIX execve is supported only on Linux and macOS')
  }
  if (cachedExecve !== undefined) return cachedExecve
  const nonblock = process.platform === 'darwin' ? 0x0004 : 0x800
  const koffi = requireKoffi()
  const libc = koffi.load(null)
  const nativeExecve = libc.func(
    'int execve(const char *pathname, const char **argv, const char **envp)',
  ) as NativeExecve
  // Darwin ARM64 passes variadic arguments differently from fixed parameters.
  const darwinFcntl = process.platform === 'darwin'
    ? libc.func('int fcntl(int fd, int cmd, ...)') as VariadicFcntl
    : undefined
  const nativeFcntl: NativeFcntl = darwinFcntl === undefined
    ? libc.func('int fcntl(int fd, int cmd, int arg)') as NativeFcntl
    : (fd, command, argument) => darwinFcntl(fd, command, 'int', argument)
  cachedExecve = (file, argv, env, control, pipeFds = []) => {
    const descriptors = control === 'pipe'
      ? [...STANDARD_FILE_DESCRIPTORS, SUBPROCESS_CONTROL_FD]
      : STANDARD_FILE_DESCRIPTORS
    for (const fd of descriptors) {
      const flags = nativeFcntl(fd, F_GETFD, 0)
      if (flags === -1) throw systemError(koffi.errno(), 'fcntl')
      if ((flags & FD_CLOEXEC) === 0) continue
      if (nativeFcntl(fd, F_SETFD, flags & ~FD_CLOEXEC) === -1) {
        throw systemError(koffi.errno(), 'fcntl')
      }
    }
    // F_SETFL changes the open file description in every process sharing it.
    // Only pipes created for this child are safe to make blocking for native targets.
    for (const fd of pipeFds) {
      const flags = nativeFcntl(fd, F_GETFL, 0)
      if (flags === -1) throw systemError(koffi.errno(), 'fcntl')
      if ((flags & nonblock) !== 0 && nativeFcntl(fd, F_SETFL, flags & ~nonblock) === -1) {
        throw systemError(koffi.errno(), 'fcntl')
      }
    }
    nativeExecve(
      file,
      [...argv, null],
      [...Object.entries(env).map(([key, value]) => `${key}=${value}`), null],
    )
    throw systemError(koffi.errno(), 'execve', file)
  }
  return cachedExecve
}

/**
 * Retain the Linux containment probe's existing binding entry.
 * @returns the cached direct POSIX execve binding.
 */
export function loadLinuxExecve(): LinuxExecve {
  return loadPosixExecve()
}
