# CoilTag abstract and proposal

- `CoilTag-Abstract-and-Proposal.pdf`: the document to send to the client.
- `proposal.html`: the editable source. Text highlighted in yellow (`class="ph"`) is a placeholder: client name, your name and contact, and the software licence terms.

To re-render the PDF after editing (needs Node and Playwright):

```bash
node render-pdf.js
```

Before sending, fill every placeholder, then delete the `.ph { background … }` rule in `proposal.html` so no highlight remains.
