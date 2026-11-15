#!/usr/bin/env node

/** Read-only Issue policy for forks without the upstream organization Project. */
import fs from 'node:fs'
import process from 'node:process'
import { pathToFileURL } from 'node:url'

import { api } from './github.mjs'
import {
  isInvalidIssueLabel,
  parseReferences,
  requiresPullRequestPolicy,
  retainIssueReferences,
  validatePullRequest,
} from './rules.mjs'

const TYPES = new Set(['Idea', 'Feature', 'Bug', 'Research', 'Task'])
const PRIORITIES = new Set(['p0', 'p1', 'p2', 'p3'])

/** Validate actual native Issue metadata; no Project status or Priority is inferred. */
export function validateNativeIssue(issue) {
  const errors = []
  const labels = issue.labels.map((label) => label.name)
  if (!TYPES.has(issue.type?.name)) errors.push('Issue must have a supported native Type')
  if (!labels.some((label) => label.startsWith('area/'))) errors.push('Issue must have an area/* label')
  if (labels.some(isInvalidIssueLabel)) errors.push('Issue has a PR-only kind or retired label')
  if (labels.filter((label) => PRIORITIES.has(label)).length > 1) errors.push('Issue has multiple Priority labels')
  if (issue.state !== 'open' && issue.state !== 'closed') errors.push('Issue must have a native open or closed state')
  if (issue.state === 'closed' && !['completed', 'not_planned'].includes(issue.state_reason)) {
    errors.push('Closed Issue must have a completed or not_planned reason')
  }
  return errors
}

/** Read live metadata using GET requests only; throw on missing or invalid Issue references. */
export async function runForkNativePolicy({ repository, eventName, event, read = api }) {
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repository)
    || event.repository?.full_name !== repository || event.repository?.fork !== true) {
    throw new Error('Fork policy requires the current fork repository identity')
  }
  const prefix = `/repos/${repository}`
  const errors = []
  if (eventName === 'issues') {
    const issue = await read(`${prefix}/issues/${event.issue.number}`)
    if (issue.pull_request) throw new Error('Issue lifecycle cannot target a pull request')
    errors.push(...validateNativeIssue(issue))
  } else if (eventName === 'pull_request' || eventName === 'pull_request_review') {
    const number = event.pull_request.number
    const pull = await read(`${prefix}/pulls/${number}`)
    const snapshot = {
      isDraft: pull.draft,
      authorType: pull.user?.type ?? 'User',
      reviewRequestCount: 0,
      reviewCount: 0,
      labels: pull.labels.map((label) => label.name),
      references: { all: [], resolving: [], related: [] },
      issues: new Map(),
    }
    if (!snapshot.isDraft && !['Bot', 'App'].includes(snapshot.authorType)) {
      const [requests, reviews] = await Promise.all([
        read(`${prefix}/pulls/${number}/requested_reviewers`),
        read(`${prefix}/pulls/${number}/reviews?per_page=100`),
      ])
      snapshot.reviewRequestCount = requests.users.length + requests.teams.length
      snapshot.reviewCount = reviews.length
    }
    if (!requiresPullRequestPolicy(snapshot)) return { eligible: false, checkedIssues: [] }
    const references = parseReferences({ body: pull.body ?? '', repository })
    for (const issueNumber of references.all) {
      const issue = await read(`${prefix}/issues/${issueNumber}`)
      if (issue.pull_request) continue
      errors.push(...validateNativeIssue(issue).map((error) => `#${issueNumber}: ${error}`))
      // Fork Priority is an explicit native label, never an invented Project value.
      const priority = issue.labels.map((label) => label.name).find((label) => PRIORITIES.has(label)) ?? null
      snapshot.issues.set(issueNumber, { priority })
    }
    snapshot.references = retainIssueReferences(references, snapshot.issues)
    errors.push(...validatePullRequest(snapshot))
    if (errors.length === 0) return { eligible: true, checkedIssues: [...snapshot.issues.keys()] }
  } else {
    throw new Error(`Unsupported fork policy event: ${eventName}`)
  }
  if (errors.length > 0) throw new Error(errors.join('\n'))
  return { eligible: true, checkedIssues: [event.issue.number] }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  Promise.resolve().then(async () => {
    const event = JSON.parse(fs.readFileSync(process.env.GITHUB_EVENT_PATH, 'utf8'))
    const result = await runForkNativePolicy({
      repository: process.env.GITHUB_REPOSITORY,
      eventName: process.env.GITHUB_EVENT_NAME,
      event,
    })
    process.stdout.write(`Fork-native Issue policy: ${result.eligible ? 'passed' : 'exempt'}; native state only, no Project writes.\n`)
  }).catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`)
    process.exitCode = 1
  })
}
