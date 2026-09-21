/* Chat page logic: sends a question, reads the streamed events, renders the answer and its sources. */
(function () {
  'use strict';

  var MAX_LENGTH = 500;
  var SESSION_KEY = 'fees-assistant-session';

  var log = document.getElementById('log');
  var empty = document.getElementById('empty');
  var status = document.getElementById('status');
  var statusText = document.getElementById('statusText');
  var form = document.getElementById('form');
  var input = document.getElementById('question');
  var sendButton = document.getElementById('send');
  var count = document.getElementById('count');
  var newChat = document.getElementById('newChat');

  var busy = false;
  var memorySession = null; // used when sessionStorage is unavailable (e.g. blocked inside an iframe)

  if (/[?&]embed=1(&|$)/.test(location.search)) document.body.classList.add('embed');

  // ---------- session ----------

  function newSessionId() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
    return 's' + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
  }

  function sessionId() {
    var id = null;
    try { id = sessionStorage.getItem(SESSION_KEY); } catch (e) { id = memorySession; }
    if (!id || !/^[A-Za-z0-9_-]{1,64}$/.test(id)) {
      id = newSessionId();
      try { sessionStorage.setItem(SESSION_KEY, id); } catch (e) { memorySession = id; }
    }
    return id;
  }

  function resetSession() {
    try { sessionStorage.removeItem(SESSION_KEY); } catch (e) { /* ignore */ }
    memorySession = null;
  }

  // ---------- DOM helpers (all text goes through textContent; only rendered markdown uses innerHTML) ----------

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function scrollDown() { log.scrollTop = log.scrollHeight; }

  function addUser(text) {
    empty.hidden = true;
    var msg = el('div', 'msg user');
    msg.appendChild(el('div', 'bubble', text));
    log.appendChild(msg);
    scrollDown();
  }

  function sourceLabel(source) {
    var pages = '';
    if (source.page && source.page_end) pages = ', pages ' + source.page + '–' + source.page_end;
    else if (source.page) pages = ', page ' + source.page;
    return source.bank + ' — ' + source.document + pages;
  }

  function addAnswer(response) {
    var msg = el('div', 'msg bot');
    var bubble = el('div', 'bubble');
    var body = el('div', 'md');
    body.innerHTML = Markdown.render(response.answer); // safe: the renderer escapes everything first
    bubble.appendChild(body);
    if (!response.grounded) bubble.appendChild(el('p', 'note', 'Not found in the documents.'));
    msg.appendChild(bubble);

    if (response.sources && response.sources.length) {
      var sources = el('div', 'sources');
      sources.setAttribute('aria-label', 'Sources');
      response.sources.forEach(function (source) { sources.appendChild(el('span', 'chip', sourceLabel(source))); });
      msg.appendChild(sources);
    }
    log.appendChild(msg);
    scrollDown();
  }

  var FRIENDLY = {
    too_many_requests: "You're asking questions quickly. Please wait a few seconds and try again.",
    busy: 'The demo is busy right now. Please try again in a moment.',
    budget_exhausted: 'The demo has reached its daily limit. Please come back tomorrow.',
    rate_limited: 'The AI service is busy right now. Please try again in a few seconds.',
    timeout: 'That took too long. Please try again.',
    unavailable: 'Something went wrong on our side. Please try again.',
    invalid: "That question can't be sent. Keep it between 1 and 500 characters.",
    network: "Can't reach the server. Check your connection and try again.",
    closed: 'The connection closed before an answer arrived. Please try again.'
  };

  function isKnownCode(code) { return Object.prototype.hasOwnProperty.call(FRIENDLY, code); }

  function addError(code, question) {
    var msg = el('div', 'msg bot error');
    var bubble = el('div', 'bubble');
    bubble.appendChild(el('div', '', isKnownCode(code) ? FRIENDLY[code] : FRIENDLY.unavailable));
    if (code !== 'invalid' && code !== 'budget_exhausted') { // retrying cannot help with these
      var retry = el('button', 'retry', 'Try again');
      retry.type = 'button';
      retry.addEventListener('click', function () { msg.remove(); ask(question, true); });
      bubble.appendChild(retry);
    }
    msg.appendChild(bubble);
    log.appendChild(msg);
    scrollDown();
  }

  function setStatus(text) {
    if (text) { statusText.textContent = text; status.hidden = false; } else { status.hidden = true; }
  }

  function setBusy(value) {
    busy = value;
    input.disabled = value;
    sendButton.disabled = value;
    if (!value) { setStatus(null); input.focus(); }
  }

  // ---------- talking to the API ----------

  function httpErrorCode(statusCode) {
    if (statusCode === 429) return 'rate_limited';
    if (statusCode === 504) return 'timeout';
    if (statusCode === 422) return 'invalid';
    return 'unavailable';
  }

  // Reads the response body as it arrives and reports each complete event.
  function readEvents(res, onEvent) {
    if (!res.body || !res.body.getReader) { // very old browsers: no streaming, parse the whole body at the end
      return res.text().then(function (text) { SSE.parseSSE(text + '\n\n').events.forEach(onEvent); });
    }
    var reader = res.body.getReader();
    var decoder = new TextDecoder('utf-8');
    var buffer = '';
    function pump() {
      return reader.read().then(function (chunk) {
        if (chunk.done) { SSE.parseSSE(buffer + '\n\n').events.forEach(onEvent); return; }
        buffer += decoder.decode(chunk.value, { stream: true });
        var parsed = SSE.parseSSE(buffer);
        buffer = parsed.rest;
        parsed.events.forEach(onEvent);
        return pump();
      });
    }
    return pump();
  }

  function ask(question, isRetry) {
    if (busy) return;
    if (!isRetry) addUser(question);
    setBusy(true);
    setStatus('Sending...');

    var outcome = null; // 'answer' or an error code, set by the events below

    fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId(), question: question })
    })
      .then(function (res) {
        if (!res.ok) {
          // the API explains a refusal in its JSON body: {code, detail}
          return res.json()
            .then(function (body) { outcome = isKnownCode(body.code) ? body.code : httpErrorCode(res.status); })
            .catch(function () { outcome = httpErrorCode(res.status); });
        }
        return readEvents(res, function (evt) {
          var data;
          try { data = JSON.parse(evt.data); } catch (e) { return; }
          if (evt.event === 'status') {
            setStatus(data.message);
          } else if (evt.event === 'answer') {
            outcome = 'answer';
            addAnswer(data.response);
          } else if (evt.event === 'error') {
            outcome = data.code || 'unavailable';
          }
        });
      })
      .catch(function () { outcome = outcome || 'network'; })
      .then(function () {
        if (outcome !== 'answer') addError(outcome || 'closed', question);
        setBusy(false);
      });
  }

  // ---------- input handling ----------

  function updateCount() { count.textContent = input.value.length + '/' + MAX_LENGTH; }

  function autoGrow() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 160) + 'px';
  }

  function submit() {
    var question = input.value.trim();
    if (!question || busy) return;
    input.value = '';
    updateCount();
    autoGrow();
    ask(question, false);
  }

  form.addEventListener('submit', function (e) { e.preventDefault(); submit(); });

  input.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(); }
  });

  input.addEventListener('input', function () { updateCount(); autoGrow(); });

  Array.prototype.forEach.call(document.querySelectorAll('.example'), function (button) {
    button.addEventListener('click', function () {
      input.value = button.getAttribute('data-question');
      submit();
    });
  });

  newChat.addEventListener('click', function () {
    if (busy) return;
    resetSession();
    Array.prototype.slice.call(log.children).forEach(function (child) { if (child !== empty) child.remove(); });
    empty.hidden = false;
    input.focus();
  });

  updateCount();
  input.focus();
})();
