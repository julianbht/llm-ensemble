# LLM Ensemble Presentations

Presentations for the LLM Ensemble project using [reveal-md](https://github.com/webpro/reveal-md).

## Setup

First time only, install dependencies:

```bash
npm install
```

Or use npx (no install needed):

```bash
npx reveal-md slides.md --watch
```

## Usage

### Live presentation mode (with hot reload)
```bash
make presentation
# or
npm start
# or
npx reveal-md slides.md --watch
```

Open http://localhost:1948 in your browser.

### Export to static HTML
```bash
make slides-export
# or
npm run export
```

Outputs to `_site/` directory.

### Export to PDF
```bash
make slides-pdf
# or
npm run pdf
```

Generates `slides.pdf`.

## reveal-md Features

- Write slides in Markdown with `---` as slide separator
- YAML frontmatter for configuration
- Code syntax highlighting
- Speaker notes with `Note:` prefix
- Vertical slides with `----` separator
- Themes: black, white, league, beige, sky, night, serif, simple, solarized
- Live reload during development

## Customization

Edit the YAML frontmatter in `slides.md`:

```yaml
---
title: Your Title
theme: black
highlightTheme: monokai
revealOptions:
  transition: 'slide'
  controls: true
  progress: true
  slideNumber: true
---
```

Available themes: `black`, `white`, `league`, `beige`, `sky`, `night`, `serif`, `simple`, `solarized`

Available transitions: `none`, `fade`, `slide`, `convex`, `concave`, `zoom`

## Keyboard Shortcuts

- **Arrow keys** or **Space**: Navigate slides
- **F**: Fullscreen
- **S**: Speaker notes view
- **O**: Overview mode
- **B** or **.**: Pause (blackout)
- **ESC**: Exit fullscreen/overview

## Resources

- [reveal.js Documentation](https://revealjs.com/)
- [reveal-md GitHub](https://github.com/webpro/reveal-md)
- [Markdown Guide](https://www.markdownguide.org/)
