# --coding: utf-8--
from llama_index.core import SQLDatabase
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    insert,
    text
)

engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()
table_name = "city_stats"
city_stat_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)

metadata_obj.create_all(engine)
sql_database = SQLDatabase(engine, include_tables=[table_name])
rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {
        "city_name": "Chicago",
        "population": 2679000,
        "country": "United States",
    },
    {"city_name": "Seoul", "population": 9776000, "country": "South Korea"},
]
for row in rows:
    stmt = insert(city_stat_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)

stmt = select(
    city_stat_table.c.city_name,
    city_stat_table.c.population,
    city_stat_table.c.country,
).select_from(city_stat_table)

with engine.connect() as connection:
    results = connection.execute(stmt).fetchall()
    for row in results:
        print(row)
print("-" * 20)
with engine.connect() as conn:
    rows = conn.execute(text("select city_name from city_stats"))
    for row in rows:
        print(row)

