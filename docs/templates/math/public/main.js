export default {
  start: () => {
    if (!document.querySelector('.math')) {
      return
    }

    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
      },
      options: {
        ignoreHtmlClass: 'mathjax-ignore',
        processHtmlClass: 'math'
      }
    }

    const script = document.createElement('script')
    script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js'
    script.async = true
    document.head.appendChild(script)
  }
}
