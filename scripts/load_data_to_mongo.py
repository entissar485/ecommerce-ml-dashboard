import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime

# Cargar variables de entorno
load_dotenv()

# Conectar a MongoDB Atlas
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('DB_NAME')

print("Conectando a MongoDB Atlas...")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Crear colección
collection = db['monthly_sales']

print("Cargando datos desde CSV...")
# Cargar el dataset agregado que creamos en el notebook
sales_monthly = pd.read_csv('data/processed/sales_monthly.csv')

print(f"Total de registros a cargar: {len(sales_monthly):,}")

# Convertir DataFrame a lista de diccionarios
records = sales_monthly.to_dict('records')

# Limpiar la colección si existe
collection.delete_many({})
print("Colección limpiada.")

# Insertar datos
print("Insertando datos en MongoDB...")
result = collection.insert_many(records)

print(f"✓ {len(result.inserted_ids):,} documentos insertados exitosamente")

# Crear índices para mejorar el rendimiento
print("Creando índices...")
collection.create_index([("product_id", 1)])
collection.create_index([("category", 1)])
collection.create_index([("year", 1), ("month", 1)])

print("✓ Índices creados")

# Verificar datos
print("\nVerificación:")
print(f"Total de documentos en la colección: {collection.count_documents({}):,}")
print("\nEjemplo de documento:")
print(collection.find_one())

client.close()
print("\n✓ Proceso completado exitosamente")