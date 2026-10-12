import asyncio
import json
from pathlib import Path
from app.ingest.pdf_parser import parse_pdf

async def main():
    pdf = list(Path('data/uploads').rglob('Политика*.pdf'))[0]
    print(f'PDF: {pdf.name}')
    # Только первые 3 страницы для быстрого теста
    result = await parse_pdf(pdf, dpi=150, max_pages=3)
    print()
    print('=== META ===')
    print(json.dumps(result['metadata'], ensure_ascii=False, indent=2))
    print()
    print('=== СТРАНИЦЫ ===')
    for page in result['pages']:
        print(f"--- Стр. {page['page']} (source={page['source']}) ---")
        for el in page['elements'][:3]:
            etype = el['type']
            if etype == 'table':
                print(f"  [table {len(el['rows'])}x{len(el['rows'][0]) if el['rows'] else 0}]")
            else:
                text = (el.get('text') or ' '.join(el.get('items', [])))[:150]
                print(f"  [{etype}] {text}")
        print()

asyncio.run(main())
