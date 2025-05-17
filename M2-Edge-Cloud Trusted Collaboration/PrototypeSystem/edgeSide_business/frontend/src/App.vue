<template>
  <div id="app">
    <router-view></router-view>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  -webkit-tap-highlight-color: transparent;
}

html {
  font-size: 16px;
}

body {
  font-family: Arial, sans-serif;
  overflow-x: hidden;
  touch-action: manipulation;
  background-color: var(--color-canvas-subtle);
}

#app {
  width: 100%;
  min-height: 100vh;
}

:root {
  --color-canvas-default: #ffffff;
  --color-canvas-subtle: #f6f8fa;
  --color-border-default: #d0d7de;
  --color-neutral-muted: #6e7781;
  --color-neutral-emphasis: #24292f;
}

@media screen and (max-width: 768px) {
  html {
    font-size: 14px;
  }
}

@media screen and (max-width: 480px) {
  html {
    font-size: 12px;
  }
}
</style> 