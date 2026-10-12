# data/inventario_julio_2025_limpio.csv

Stock snapshot of Display Internacional SRL, exported July 2025 and cleaned (duplicates removed,
hidden negative stock recovered, empty location column dropped, descriptions normalised).
2,098 rows, one per SKU.

Target: `dead = (Stock <= 0) & (Punto reorden == 0)` on `Tipo == 'Producto'` rows with `Costo > 0`.

| Column | Meaning | Use in model |
|---|---|---|
| Código | SKU code; alphabetic prefix often encodes a product line | feature (prefix) |
| Descripción | free-text description, Spanish with English/abbrev mix | feature (parsed) |
| Tipo | Producto / Servicio (keyword-derived) | filter: Producto only |
| Stock | units on hand (negative = sold more than received) | **excluded — the label is built from it** |
| Estado stock | Negativo / Sin stock / Bajo mínimo / En stock | **excluded — derived from Stock** |
| Costo | average unit cost, DOP (assumed) | not a feature (log(Costo) tried, no gain); prices a false positive |
| Precio detalle | retail price | **excluded as feature** — used only in the cost model (FP cost = 2 × margin) |
| Precio mayor 1/2/3 | wholesale tiers | **excluded as feature** (markup × cost) |
| Precio mínimo | floor price | **excluded as feature** (markup × cost) |
| Margen % | (Precio − Costo)/Precio | **excluded as feature** (function of price and cost) |
| Valor inventario | Stock × Costo | **excluded — derived from Stock** |
| Empaque | units per pack | not a feature (1 for 99.7% of SKUs) |
| Suplidor (ID) | supplier code (no names available) | feature (freq-encoded) |
| Área (ID) | area code | feature (freq-encoded) |
| Dpto (ID) | department code | feature (freq + top-N one-hot) |
| Punto reorden / Cant. reorden | reorder point / order qty | **excluded — the label is built from Punto reorden** |
| Alertas | data-quality flags | **excluded — derived from stock/reorder columns** |
