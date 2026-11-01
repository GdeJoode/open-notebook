# Leest de inloggegevens uit de Docker secrets en start SurrealDB (rocksdb).
# Bij een leeg volume maakt SurrealDB met deze gegevens de eerste rootgebruiker.
# Witruimte aan het eind (ook \r uit een Windows-bestand) telt niet mee, net als
# in services/open-notebook/start.py.
set -eu
SURREAL_USER="$(/busybox tr -d '\r\n' </run/secrets/surreal_gebruiker)"
SURREAL_PASS="$(/busybox tr -d '\r\n' </run/secrets/surreal_wachtwoord)"
export SURREAL_USER SURREAL_PASS
exec /surreal start --log info rocksdb:/data/database.db
