# Guía Paso a Paso - Levantar la Aplicación Web

```bash
cd app
docker-compose build
docker-compose up -d
sleep 30
curl http://localhost:8000/health
```

Debes ver: `"modelo_cargado": true`

---

## URLs

- **Interfaz Web:** http://localhost:7860
- **API:** http://localhost:8000
- **Documentación:** http://localhost:8000/docs

---

## Comandos Útiles

```bash
# Ver logs
docker-compose logs -f

# Reiniciar
docker-compose restart

# Detener
docker-compose down

# Reconstruir (si es que hay cambios)
docker-compose build --no-cache
docker-compose up -d
```

---

## Si algo falla

```bash
# Ver logs del backend
docker-compose logs backend | tail -30

# Verificar health
curl http://localhost:8000/health

# Reiniciar todo
docker-compose down
docker-compose up -d
```

---

**Abrir http://localhost:7860 y usar los IDs de los datos para predecir.**


















