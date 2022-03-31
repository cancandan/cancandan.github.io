
/* <!-- register service worker --> */
if ("serviceWorker" in navigator) {
  navigator.serviceWorker
    .register("/service-worker.js", { scope: "/" })
    .catch(function (error) {
      console.log("Service worker failed to register!", error);
    });
}



/* <!--
  Removes extra leading spaces from the given code. If the given code has 4
  lines and all 4 lines have more than 2 leading spaces, then it will remove
  first 2 spaces from all the lines.
--> */
function normalizeWhiteSpace(code) {
  let lines = code.split("\n");
  if (lines.length < 1) {
    return code;
  }

  let spacesToRemove = Infinity;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].length < 1) {
      continue;
    }

    spacesToRemove = Math.min(spacesToRemove, lines[i].search(/\S|$/));
  }

  return lines
    .map(function (v) {
      return v.substring(spacesToRemove);
    })
    .join("\n");
}

function createCodeElement(code, language) {
  let element = document.createElement("code");
  element.innerHTML = normalizeWhiteSpace(code);
  if (language) {
    element.classList.add(`language-${language}`);
  }

  return element;
}

function createFileLinkElement(url) {
  let element = document.createElement("a");
  element.innerHTML = "View file";
  element.href = url;

  let container = document.createElement("div");
  container.classList.add("view-file");
  container.appendChild(element);
  return container;
}

/* <!--
  Searches for `pre` elements with `data-src` attribute. For each such `pre`
  element, it attempts to load the content at the URL provided using `data-src`.
  It populates the `pre` element with a child `code` element containing the
  loaded content. It also adds an anchor `a` element to the `pre` container with
  a link to view the original resource. The following optional attributes can be
  used to control its behavior.

  1. `data-view`: URL at which 'viewable' version of the resource can be found,
     e.g., GitHub BLOB URL. Defaults to the URL provided using `data-src`.
  2. `data-start`: Starting line number in the resource to slice the displayed
     content. Indexed starting from position 1. Default: 1
  3. `data-end`: Ending line number in the resource to slice the displayed
     content. Indexed starting from position 1. Default: -1 (end)
  4. `data-lang`: Language hinting, passed to 'highlight.js' for syntax
     highlighting.
--> */
addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("pre[data-src]").forEach(function (item) {
    let url = item.getAttribute("data-src");
    let viewURL = item.getAttribute("data-view") || url;
    let start = parseInt(item.getAttribute("data-start") || 1);
    let end = parseInt(item.getAttribute("data-end") || -1);
    let language = item.getAttribute("data-lang");

    let xhr = new XMLHttpRequest();
    xhr.open("GET", url);
    xhr.addEventListener("load", function () {
      let code = `error: unable to load the resource at "${url}"`;
      if (this.responseText) {
        code = this.responseText
          .split("\n")
          .slice(start - 1, end)
          .join("\n");
      }

      let codeElement = createCodeElement(code, language);
      hljs.highlightBlock(codeElement);
      item.appendChild(codeElement);
      item.appendChild(createFileLinkElement(viewURL));
    });

    xhr.send();
  });
});
