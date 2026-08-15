/* Mermaid initialisation for the mkdocs (readthedocs) theme.
 *
 * pymdownx.superfences with `fence_div_format` renders ```mermaid blocks as
 * `<div class="mermaid">...</div>`. This script initialises Mermaid and
 * renders every such block after the DOM is ready.
 */
(function () {
  "use strict";

  function init() {
    if (!window.mermaid) {
      return;
    }
    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: "loose",
      theme: "default",
    });
    // Render all diagrams currently in the DOM.
    window.mermaid.run({ querySelector: ".mermaid" });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
