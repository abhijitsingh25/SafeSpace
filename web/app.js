/* SafeSpace chat client.
 *
 * Talks to /ask/stream over SSE. EventSource only does GET, so this reads the POST
 * response body as a stream and parses the event frames itself.
 */
(function () {
  "use strict";

  var API = (window.SAFESPACE_API || "").replace(/\/$/, "");
  var SESSION_KEY = "safespace.session";
  var REGION_KEY = "safespace.region";

  var messagesEl = document.getElementById("messages");
  var formEl = document.getElementById("composer");
  var inputEl = document.getElementById("input");
  var sendEl = document.getElementById("send");
  var regionEl = document.getElementById("region");
  var helpBtn = document.getElementById("help-btn");
  var resetBtn = document.getElementById("reset-btn");
  var dialogEl = document.getElementById("help-dialog");
  var helpBodyEl = document.getElementById("help-body");

  // --- storage (may throw in private mode; never let it break the page) -------
  function store(key, value) {
    try {
      if (value === undefined) return window.localStorage.getItem(key);
      window.localStorage.setItem(key, value);
    } catch (e) {
      return null;
    }
  }

  function newSessionId() {
    if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
    return String(Date.now()) + Math.random().toString(16).slice(2);
  }

  var sessionId = store(SESSION_KEY) || newSessionId();
  store(SESSION_KEY, sessionId);

  var savedRegion = store(REGION_KEY);
  if (savedRegion) regionEl.value = savedRegion;
  regionEl.addEventListener("change", function () {
    store(REGION_KEY, regionEl.value);
  });

  // --- rendering --------------------------------------------------------------
  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /* Minimal markdown for the subset the backend emits: bold, links, bullet lists,
     rules, and paragraphs. Input is escaped first, so model output cannot inject HTML. */
  function renderMarkdown(raw) {
    var safe = escapeHtml(raw);
    var blocks = safe.split(/\n{2,}/);
    var html = "";

    blocks.forEach(function (block) {
      block = block.trim();
      if (!block) return;

      if (/^(-{3,}|_{3,})$/.test(block)) {
        html += "<hr />";
        return;
      }

      var lines = block.split("\n");
      var isList = lines.every(function (line) {
        return /^\s*[-*]\s+/.test(line) || !line.trim();
      });

      if (isList) {
        html += "<ul>";
        lines.forEach(function (line) {
          if (!line.trim()) return;
          html += "<li>" + inline(line.replace(/^\s*[-*]\s+/, "")) + "</li>";
        });
        html += "</ul>";
      } else {
        html += "<p>" + inline(block).replace(/\n/g, "<br />") + "</p>";
      }
    });

    return html;
  }

  function inline(text) {
    return text
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/_([^_\n]+)_/g, "<em>$1</em>")
      .replace(/\bhttps?:\/\/[^\s<)]+/g, function (url) {
        return '<a href="' + url + '" target="_blank" rel="noopener noreferrer">' + url + "</a>";
      });
  }

  function addMessage(role, html, options) {
    options = options || {};
    var wrapper = document.createElement("div");
    wrapper.className = "msg " + role + (options.crisis ? " crisis" : "");

    var bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = html;
    wrapper.appendChild(bubble);

    if (options.meta) {
      var meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = options.meta;
      bubble.appendChild(meta);
    }

    messagesEl.appendChild(wrapper);
    wrapper.scrollIntoView({ behavior: "smooth", block: "end" });
    return bubble;
  }

  function addTyping() {
    return addMessage(
      "assistant",
      '<div class="typing"><span></span><span></span><span></span></div>'
    );
  }

  // --- networking -------------------------------------------------------------
  function parseFrame(frame) {
    var event = "message";
    var data = "";
    frame.split("\n").forEach(function (line) {
      if (line.indexOf("event:") === 0) event = line.slice(6).trim();
      else if (line.indexOf("data:") === 0) data += line.slice(5).trim();
    });
    if (!data) return null;
    try {
      return { event: event, data: JSON.parse(data) };
    } catch (e) {
      return null;
    }
  }

  function setBusy(busy) {
    sendEl.disabled = busy;
    inputEl.disabled = busy;
    if (!busy) inputEl.focus();
  }

  async function send(message) {
    addMessage("user", renderMarkdown(message));
    setBusy(true);
    var bubble = addTyping();

    try {
      var response = await fetch(API + "/ask/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: message,
          session_id: sessionId,
          region: regionEl.value
        })
      });

      if (response.status === 429) {
        bubble.innerHTML = renderMarkdown(
          "That's a lot of messages at once — take a breath, I'm still here. Try again in a moment."
        );
        return;
      }
      if (!response.ok || !response.body) {
        throw new Error("HTTP " + response.status);
      }

      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";
      var toolUsed = null;
      var delivered = false;

      while (true) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });

        var parts = buffer.split("\n\n");
        buffer = parts.pop();

        for (var i = 0; i < parts.length; i++) {
          var frame = parseFrame(parts[i]);
          if (!frame) continue;

          if (frame.event === "session" && frame.data.session_id) {
            sessionId = frame.data.session_id;
            store(SESSION_KEY, sessionId);
          } else if (frame.event === "tool") {
            toolUsed = frame.data.tool;
          } else if (frame.event === "error") {
            bubble.innerHTML = renderMarkdown(frame.data.detail || "Something went wrong.");
            delivered = true;
          } else if (frame.event === "final") {
            bubble.innerHTML = renderMarkdown(frame.data.response);
            if (frame.data.crisis) {
              bubble.parentElement.className = "msg assistant crisis";
            } else if (toolUsed) {
              var meta = document.createElement("div");
              meta.className = "meta";
              meta.textContent = "via " + toolUsed.replace(/_/g, " ");
              bubble.appendChild(meta);
            }
            delivered = true;
          }
        }
      }

      if (!delivered) {
        bubble.innerHTML = renderMarkdown("I didn't catch that — could you say it again?");
      }
    } catch (error) {
      bubble.innerHTML = renderMarkdown(
        "I can't reach SafeSpace right now. That's a problem on my end, not anything " +
          "you did. Please try again in a moment.\n\nIf you need someone urgently, use " +
          "the **Need help now** button above."
      );
    } finally {
      setBusy(false);
    }
  }

  // --- events -----------------------------------------------------------------
  formEl.addEventListener("submit", function (event) {
    event.preventDefault();
    var message = inputEl.value.trim();
    if (!message) return;
    inputEl.value = "";
    inputEl.style.height = "auto";
    send(message);
  });

  inputEl.addEventListener("input", function () {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
  });

  inputEl.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      formEl.requestSubmit();
    }
  });

  resetBtn.addEventListener("click", function () {
    sessionId = newSessionId();
    store(SESSION_KEY, sessionId);
    messagesEl.innerHTML = "";
    addMessage("assistant", renderMarkdown("Fresh start. What's on your mind?"));
  });

  helpBtn.addEventListener("click", async function () {
    helpBodyEl.textContent = "Loading…";
    if (typeof dialogEl.showModal === "function") dialogEl.showModal();

    try {
      var response = await fetch(
        API + "/crisis-resources?region=" + encodeURIComponent(regionEl.value)
      );
      var data = await response.json();
      helpBodyEl.innerHTML = (data.lines || [])
        .map(function (line) {
          return (
            '<div class="line"><strong>' +
            escapeHtml(line.name) +
            "</strong><span>" +
            inline(escapeHtml(line.contact)) +
            "</span></div>"
          );
        })
        .join("");
    } catch (e) {
      // The hotline panel must work even when the API does not.
      helpBodyEl.innerHTML =
        '<div class="line"><strong>Find a helpline in your country</strong>' +
        '<span><a href="https://findahelpline.com" target="_blank" ' +
        'rel="noopener noreferrer">findahelpline.com</a></span></div>' +
        '<div class="line"><strong>Or call your local emergency number.</strong></div>';
    }
  });

  inputEl.focus();
})();
