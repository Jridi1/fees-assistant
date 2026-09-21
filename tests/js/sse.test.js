const test = require('node:test');
const assert = require('node:assert');
const { parseSSE } = require('../../feesbot/static/sse.js');

test('parses complete events', () => {
  const { events, rest } = parseSSE('event: status\ndata: {"a":1}\n\nevent: answer\ndata: {"b":2}\n\n');
  assert.deepStrictEqual(events, [
    { event: 'status', data: '{"a":1}' },
    { event: 'answer', data: '{"b":2}' },
  ]);
  assert.strictEqual(rest, '');
});

test('keeps an unfinished event as the remainder and completes it with the next chunk', () => {
  const first = parseSSE('event: status\ndata: {"a":1}\n\nevent: answ');
  assert.strictEqual(first.events.length, 1);
  assert.strictEqual(first.rest, 'event: answ');

  const second = parseSSE(first.rest + 'er\ndata: {"b":2}\n\n');
  assert.deepStrictEqual(second.events, [{ event: 'answer', data: '{"b":2}' }]);
});

test('a chunk boundary in the middle of the blank line separator is handled', () => {
  const first = parseSSE('event: status\ndata: x\n');
  assert.deepStrictEqual(first.events, []);
  const second = parseSSE(first.rest + '\n');
  assert.deepStrictEqual(second.events, [{ event: 'status', data: 'x' }]);
});

test('accepts CRLF line endings', () => {
  const { events } = parseSSE('event: status\r\ndata: hi\r\n\r\n');
  assert.deepStrictEqual(events, [{ event: 'status', data: 'hi' }]);
});

test('joins multiple data lines and ignores comments', () => {
  const { events } = parseSSE(': keep-alive\nevent: answer\ndata: line1\ndata: line2\n\n');
  assert.deepStrictEqual(events, [{ event: 'answer', data: 'line1\nline2' }]);
});

test('events without data are dropped and the default event name is "message"', () => {
  assert.deepStrictEqual(parseSSE('event: status\n\n').events, []);
  assert.deepStrictEqual(parseSSE('data: hello\n\n').events, [{ event: 'message', data: 'hello' }]);
});

test('a colon inside the data value is preserved', () => {
  const { events } = parseSSE('event: status\ndata: {"message":"a: b"}\n\n');
  assert.strictEqual(events[0].data, '{"message":"a: b"}');
});
