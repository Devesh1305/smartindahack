const { chromium } = require('playwright');
const S=__dirname + '/';
(async()=>{const b=await chromium.launch();const p=await b.newPage();
await p.goto('file://'+S+'proposal.html',{waitUntil:'networkidle'});await p.waitForTimeout(1500);
await p.pdf({path:S+'CoilTag-Abstract-and-Proposal.pdf',format:'A4',printBackground:true,displayHeaderFooter:true,
 headerTemplate:'<span></span>',
 footerTemplate:'<div style="font-size:8px;width:100%;padding:0 18mm;display:flex;justify-content:space-between;color:#4B5866;font-family:Arial"><span>CoilTag · Abstract and proposal · CT-2026-001</span><span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span></div>',
 margin:{top:'20mm',bottom:'22mm',left:'18mm',right:'18mm'}});
await b.close();})();
