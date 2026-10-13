# Raw OMIE file bytes for tests, so they never hit the network

from datetime import date


def marginalpdbc_file(day: date, prices: list[float]) -> bytes:
    lines = ["MARGINALPDBC;"]
    for period, price in enumerate(prices, start=1):
        lines.append(f"{day.year};{day.month:02d};{day.day:02d};{period};{price};{price};")
    lines.append("*")
    return "\n".join(lines).encode("latin-1")


def curva_pbc_file(day: date, steps: list[tuple[str, str, str]]) -> bytes:
    # steps are (side, quantity, price), euro formatted
    header = (
        "Periodo;Fecha;Pais;Unidad;Tipo Oferta;Potencia Compra/Venta;"
        "Precio Compra/Venta;Ofertada (O)/Casada (C);Tipología de Oferta;"
    )
    lines = [f"OMIE - Mercado de electricidad;Fecha Emisión :{day:%d/%m/%Y};;;;;;;;", "", header]
    for side, quantity, price in steps:
        lines.append(f"H1Q1;{day:%d/%m/%Y};MI;;{side};{quantity};{price};O;S;")
    lines.append("*")
    return "\n".join(lines).encode("latin-1")
