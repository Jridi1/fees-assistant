const test = require('node:test');
const assert = require('node:assert');
const { render } = require('../../feesbot/static/markdown.js');

// ---- security: nothing the model writes may become active HTML ----

test('HTML tags are escaped, never rendered', () => {
  const html = render('<script>alert(1)</script>');
  assert.ok(!html.includes('<script>'));
  assert.ok(html.includes('&lt;script&gt;'));
});

test('event-handler attributes cannot break out', () => {
  const html = render('<img src=x onerror=alert(1)> and "quoted" \'single\'');
  assert.ok(!/<img/i.test(html));
  assert.ok(html.includes('&lt;img'));
  assert.ok(html.includes('&quot;quoted&quot;'));
});

test('links are not supported, so javascript: URLs cannot appear', () => {
  const html = render('[click me](javascript:alert(1)) and https://example.com');
  assert.ok(!/<a[\s>]/i.test(html));
  assert.ok(!/href/i.test(html));
});

test('markup smuggled inside table cells and headings is escaped too', () => {
  const html = render('# <b>x</b>\n\n| a | b |\n|---|---|\n| <i>y</i> | `<u>z</u>` |');
  assert.ok(!/<b>|<i>|<u>/.test(html));
  assert.ok(html.includes('&lt;i&gt;y&lt;/i&gt;'));
});

// ---- formatting ----

test('bold, italic and code', () => {
  const html = render('**€10** and *note* and `x**y**`');
  assert.ok(html.includes('<strong>€10</strong>'));
  assert.ok(html.includes('<em>note</em>'));
  assert.ok(html.includes('<code>x**y**</code>')); // no bold inside code
});

test('a lone asterisk in arithmetic is not italic', () => {
  assert.ok(!render('5 * 3 * 2').includes('<em>'));
});

test('headings are shifted down to stay small inside a bubble', () => {
  assert.ok(render('# Title').includes('<h3>Title</h3>'));
  assert.ok(render('### Sub').includes('<h5>Sub</h5>'));
  assert.ok(render('###### Deep').includes('<h6>Deep</h6>'));
});

test('bullet and numbered lists', () => {
  assert.strictEqual(render('- one\n- two'), '<ul><li>one</li><li>two</li></ul>');
  assert.strictEqual(render('1. first\n2. second'), '<ol><li>first</li><li>second</li></ol>');
});

test('a list ends at a blank line and a different list type starts a new list', () => {
  assert.strictEqual(render('- a\n\n1. b'), '<ul><li>a</li></ul><ol><li>b</li></ol>');
});

test('paragraphs keep single line breaks', () => {
  assert.strictEqual(render('line one\nline two\n\nnext'), '<p>line one<br>line two</p><p>next</p>');
});

test('horizontal rule', () => {
  assert.ok(render('above\n\n---\n\nbelow').includes('<hr>'));
});

test('a table like the ones the assistant writes', () => {
  const md = [
    '| Card type | Standard fee | Express |',
    '|-----------|--------------|---------|',
    '| **N26 Standard** | **€10.00** | €30.00 |',
    '| N26 Metal | €45.00 | €65.00 |',
  ].join('\n');
  const html = render(md);
  assert.ok(html.startsWith('<div class="table-wrap"><table>'));
  assert.ok(html.includes('<th>Card type</th>'));
  assert.ok(html.includes('<td><strong>N26 Standard</strong></td>'));
  assert.strictEqual((html.match(/<tr>/g) || []).length, 3); // header + 2 rows
});

test('short table rows are padded with empty cells', () => {
  const html = render('| a | b |\n|---|---|\n| only |');
  assert.ok(html.includes('<td>only</td><td></td>'));
});

test('a pipe line that is not followed by a separator is plain text', () => {
  assert.ok(!render('| just | text |').includes('<table>'));
});

test('empty input renders nothing', () => {
  assert.strictEqual(render(''), '');
});
