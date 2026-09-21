/*
 * Small, safe Markdown renderer for chat answers.
 *
 * Safety: the whole text is HTML-escaped FIRST, and only then are a few known patterns turned into
 * tags. So nothing the model (or a document it read) writes can become a script, an image, an event
 * handler or a link. Links are deliberately not supported.
 *
 * Supported: #-headings, **bold**, *italic*, `code`, - / 1. lists, | tables |, --- rules, paragraphs.
 * Works in the browser (window.Markdown) and in Node (module.exports).
 */
(function (root) {
  'use strict';

  function escapeHtml(text) {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // Inline formatting on text that is already escaped. Code spans are lifted out first so their
  // contents are not formatted, then put back.
  function inline(escaped) {
    var spans = [];
    var text = escaped.replace(/`([^`]+)`/g, function (_, code) {
      spans.push('<code>' + code + '</code>');
      return '\u0000' + (spans.length - 1) + '\u0000';
    });
    text = text.replace(/\*\*([^*]+?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?!\*)/g, '$1<em>$2</em>');
    return text.replace(/\u0000(\d+)\u0000/g, function (_, i) { return spans[Number(i)]; });
  }

  function splitRow(line) {
    var trimmed = line.trim().replace(/^\|/, '').replace(/\|$/, '');
    return trimmed.split('|').map(function (cell) { return cell.trim(); });
  }

  var TABLE_SEPARATOR = /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/;

  function renderTable(header, rows) {
    var html = '<div class="table-wrap"><table><thead><tr>';
    header.forEach(function (cell) { html += '<th>' + inline(escapeHtml(cell)) + '</th>'; });
    html += '</tr></thead><tbody>';
    rows.forEach(function (row) {
      html += '<tr>';
      header.forEach(function (_, i) { html += '<td>' + inline(escapeHtml(row[i] || '')) + '</td>'; });
      html += '</tr>';
    });
    return html + '</tbody></table></div>';
  }

  function render(markdown) {
    var lines = String(markdown).replace(/\r\n/g, '\n').split('\n');
    var out = [];
    var paragraph = [];
    var list = null; // { tag: 'ul' | 'ol', items: [] }

    function flushParagraph() {
      if (paragraph.length) {
        out.push('<p>' + paragraph.map(function (l) { return inline(escapeHtml(l)); }).join('<br>') + '</p>');
        paragraph = [];
      }
    }
    function flushList() {
      if (list) {
        out.push('<' + list.tag + '>' + list.items.map(function (i) { return '<li>' + i + '</li>'; }).join('') + '</' + list.tag + '>');
        list = null;
      }
    }

    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      var match;

      if (/^\s*$/.test(line)) { flushParagraph(); flushList(); continue; }

      if (/^\s*\|/.test(line) && i + 1 < lines.length && TABLE_SEPARATOR.test(lines[i + 1])) {
        flushParagraph(); flushList();
        var header = splitRow(line);
        var rows = [];
        i += 2;
        while (i < lines.length && /^\s*\|/.test(lines[i])) { rows.push(splitRow(lines[i])); i++; }
        i--;
        out.push(renderTable(header, rows));
        continue;
      }

      if ((match = /^(#{1,6})\s+(.*)$/.exec(line))) {
        flushParagraph(); flushList();
        var level = Math.min(match[1].length + 2, 6); // keep headings small inside a chat bubble
        out.push('<h' + level + '>' + inline(escapeHtml(match[2].trim())) + '</h' + level + '>');
        continue;
      }

      if (/^\s*([-*_])\1{2,}\s*$/.test(line)) { flushParagraph(); flushList(); out.push('<hr>'); continue; }

      if ((match = /^\s*[-*+]\s+(.*)$/.exec(line))) {
        flushParagraph();
        if (!list || list.tag !== 'ul') { flushList(); list = { tag: 'ul', items: [] }; }
        list.items.push(inline(escapeHtml(match[1])));
        continue;
      }
      if ((match = /^\s*\d+[.)]\s+(.*)$/.exec(line))) {
        flushParagraph();
        if (!list || list.tag !== 'ol') { flushList(); list = { tag: 'ol', items: [] }; }
        list.items.push(inline(escapeHtml(match[1])));
        continue;
      }

      flushList();
      paragraph.push(line);
    }
    flushParagraph();
    flushList();
    return out.join('');
  }

  var api = { render: render, escapeHtml: escapeHtml };
  root.Markdown = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
