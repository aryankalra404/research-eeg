import mammoth, sys, re
style_map = """
p[style-name='Heading 1'] => h1:fresh
p[style-name='Heading 2'] => h2:fresh
"""
with open("STEW_Results.docx","rb") as f:
    res = mammoth.convert_to_html(f, style_map=style_map)
html = res.value
css = """
@page { size: A4; margin: 18mm 20mm 18mm 20mm;
  @bottom-center { content: counter(page); font: 9pt Calibri, Carlito, sans-serif; color:#6B7C93; } }
body { font-family: Calibri, Carlito, 'Liberation Sans', Arial, sans-serif; font-size: 10pt; color:#111; line-height:1.35; }
h1 { font-size: 14pt; color:#1F3A5F; margin: 16pt 0 6pt; break-after: avoid; }
p:first-child strong, p:first-child { }
table { border-collapse: collapse; width: 100%; margin: 2pt 0 8pt; font-size: 8pt; break-inside: avoid; }
td, th { border: 0.6pt solid #B8C4D2; padding: 2.5pt 5pt; text-align: center; vertical-align: middle; }
td:first-child, th:first-child { text-align: left; }
thead td, thead th { font-weight:bold; } thead td, thead th { background:#E8EEF5; color:#1F3A5F; font-weight:bold; }
td p { margin: 0; }
img { display:block; margin: 6pt auto 2pt; max-width: 100%; }
p { margin: 0 0 5pt; }
ul { margin: 2pt 0 8pt; padding-left: 16pt; } li { margin-bottom: 2pt; }
.title { font-size: 17pt; font-weight: bold; color:#1F3A5F; margin-bottom: 2pt; }
.sub { font-style: italic; color:#555; margin-bottom: 12pt; }
p:has(> img) { break-after: avoid; break-inside: avoid; } .keep { break-inside: avoid; } .keep.big { break-inside: auto; } .keep > p { break-after: avoid; margin-top: 8pt; }
"""
# first two paragraphs = title + subtitle
html = html.replace("<p>", '<p class="title">', 1)
i = html.index("</p>") + 4
html = html[:i] + html[i:].replace("<p>", '<p class="sub">', 1)
parts = html.split("<table>")
out = parts[0]
for k, part in enumerate(parts[1:]):
    j = out.rfind("<p")
    cap = out[j:]
    big = "Table 2." in cap
    if j >= 0 and re.search(r"Table \d+\.", cap) and cap.endswith("</p>"):
        out = out[:j] + ('<div class="keep big">' if big else '<div class="keep">') + cap
    else:
        out += '<div class="keep">'
    end = part.index("</table>") + len("</table>")
    out += "<table>" + part[:end] + "</div>" + part[end:]
html = out
html = html.replace('<h1>2. Baseline', '<h1 style="break-before: page">2. Baseline', 1)
html = html.replace('<h1>6. Proposed', '<h1 style="break-before: page">6. Proposed', 1)
open("STEW_Results.html","w").write(f"<!doctype html><html><head><meta charset='utf-8'><style>{css} .keep.big thead{{display:table-header-group}} .keep.big tr{{break-inside:avoid}}</style></head><body>{html}</body></html>")
print(res.messages[:5])
