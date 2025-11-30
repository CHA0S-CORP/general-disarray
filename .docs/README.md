# Documentation Structure for readme.io

## Categories

1. **Overview**
   - index.md (Overview)

2. **Setup**
   - getting-started.md (Getting Started)
   - configuration.md (Configuration)

3. **API**
   - api-reference.md (API Reference)

4. **Features**
   - tools.md (Built-in Tools)

5. **Development**
   - plugins.md (Creating Plugins)

6. **Guides**
   - examples.md (Examples)

## Import Instructions for readme.io

1. Create a new project in readme.io
2. Go to Documentation > Guides
3. Create categories matching the structure above
4. For each markdown file:
   - Create a new page in the appropriate category
   - Copy the content (excluding the frontmatter)
   - Set the title and slug from the frontmatter
5. Configure navigation order in readme.io dashboard

## Frontmatter Format

Each file includes readme.io-compatible frontmatter:

```yaml
---
title: "Page Title"
excerpt: "Short description"
category: "Category Name"
slug: "url-slug"
---
```

## API Reference

For the API Reference, consider using readme.io's API Reference feature:

1. Export an OpenAPI/Swagger spec from FastAPI:
   ```bash
   curl http://localhost:8080/openapi.json > openapi.json
   ```

2. Import into readme.io API Reference section

3. This provides interactive API documentation with try-it-now features
