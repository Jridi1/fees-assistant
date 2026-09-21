/* Server-Sent Events parser. Works in the browser (window.SSE) and in Node (module.exports). */
(function (root) {
  'use strict';

  /**
   * Split a text buffer into complete SSE events plus the unfinished remainder.
   * Network chunks can cut an event anywhere, so callers keep `rest` and prepend it to the next chunk.
   * Returns { events: [{ event: string, data: string }], rest: string }.
   */
  function parseSSE(buffer) {
    var normalized = buffer.replace(/\r\n/g, '\n');
    var blocks = normalized.split('\n\n');
    var rest = blocks.pop(); // the last piece is incomplete (or empty)
    var events = [];
    blocks.forEach(function (block) {
      var name = 'message';
      var data = [];
      block.split('\n').forEach(function (line) {
        if (line.indexOf(':') === 0) return; // comment / keep-alive
        var colon = line.indexOf(':');
        var field = colon === -1 ? line : line.slice(0, colon);
        var value = colon === -1 ? '' : line.slice(colon + 1).replace(/^ /, '');
        if (field === 'event') name = value;
        else if (field === 'data') data.push(value);
      });
      if (data.length) events.push({ event: name, data: data.join('\n') });
    });
    return { events: events, rest: rest };
  }

  var api = { parseSSE: parseSSE };
  root.SSE = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
