# ADR 0002: Afscherming van de API, de MCP-routes en de Compose-omgeving

- **Status:** besloten
- **Datum:** 2026-10-29
- **Raakt FO-paragraaf:** 9 (toegang, externe uitvraag), 10 (MCP-server), 11 (Langflow als MCP-server uit), 12.1, 13.2 punt 3, 14 (MCP-server)
- **Lost op:** verschilanalyse conflict 5, 6 en 12, open vraag 1 en 2

## Context

Harde eis 1 wordt op drie plekken afgedwongen: het Docker-netwerk, de modelkeuze in de Langflow-componenten en het filter in de API (Application Programming Interface) van de fork (FO 9). De MCP-server (Model Context Protocol) is open-notebook-mcp, via de REST-API van de fork; het filter op intern zit in die API (FO 10, FO 14). Elke gebruiker en elke agent krijgt een eigen sleutel met rechten lezen extern, lezen intern of beheren. API-sleutels van cloudaanbieders worden versleuteld opgeslagen. Alleen de Langflow-editor en de MCP-server zijn bereikbaar, op localhost of via Tailscale. Geheimen lopen via Docker secrets (FO 9, FO 12.1).

In de fork is elk van die plekken te omzeilen:

- `PasswordAuthMiddleware` laat alles door als `OPEN_NOTEBOOK_PASSWORD` leeg is (`apps/app-main/src/app_main/api/auth.py:28-29`);
- uitgezonderd van het wachtwoord zijn `/api/config`, `/api/auth/status`, `/api/v1/agents/` (`apps/app-main/src/app_main/api/app.py:280-292`) en alle routes `/api/sources/*/pdf` (`auth.py:54-58`);
- sleutels met rechten (`require_agent_key`, `apps/app-main/src/app_main/api/agent_auth.py:46-101`; tabel `agent_keys`, `migrations/77.surrealql`) gelden alleen voor `api/routers/agents.py`, met rechten `read`/`write`/`admin` die niets met classificatie te maken hebben;
- geen andere router heeft een eigen authenticatie (`app.py:348-418`), en geen leesroute filtert op classificatie (verschilanalyse 3.8);
- de extractiedienst neemt een willekeurig pad aan in `POST /extract` (`services/extraction/api.py:634-641`). Hij publiceert poort 8103 (`docker-compose.yml:147-148`), en `services_proxy` stuurt verzoeken door (`apps/app-main/src/app_main/api/routers/services_proxy.py:134`). PC.7 is niet gestart;
- `surrealdb-mcp` (`packages/surrealdb-service/src/surrealdb_service/mcp/server.py`) praat direct met de database, heeft `query_database` voor willekeurige SurrealQL, schrijftools en geen authenticatie (`:11-17`, `:77-83`);
- twee andere MCP-servers: `packages/llm-manager/src/llm_manager/mcp/server.py` (modelbeheer en gebruik) en `packages/file-manager/src/file_manager/mcp/server.py`, met `read_file_content` op een opgegeven pad (`:87`);
- er is één netwerk en alle poorten zijn gepubliceerd, ook SurrealDB op 8000 (`docker-compose.yml:1-4`, `:10-11`);
- de database-inloggegevens `root`/`root` staan in platte tekst (`docker-compose.yml:24`, `:165-166`);
- tokens en sleutels lopen via omgevingsvariabelen (`:140`, `:172`; `packages/llm-manager/src/llm_manager/providers/registry.py`, velden `api_key_env_var`).

**Gedrag van open-notebook-mcp (gecontroleerd).** Het pakket is van Epochal-dev, staat onder de MIT-licentie en is geen project van lfnovo. Uit `src/open_notebook_mcp/server.py`:

- het stuurt `Authorization: Bearer <OPEN_NOTEBOOK_PASSWORD>` als die variabele gezet is;
- het roept `GET`/`POST`/`PUT`/`DELETE` aan op `/api/notebooks`, `/api/sources` en `/api/notes`, `POST /api/search`, `/api/search/ask`, `/api/search/ask/simple`, `/api/models`, `/api/chat/*` en `/api/settings`;
- de zoektool stuurt alleen `query`, `type`, `limit` en eventueel `notebook_id`, dus geen `search_notes`;
- er is geen alleen-lezen-modus of lijst met toegestane tools;
- standaard draait het via stdio, en via `MCP_TRANSPORT=streamable-http` via HTTP.

Aanname A2 klopt daarmee voor authenticatie en paden. Of de antwoorden van de fork precies de vorm hebben die het pakket verwacht, is niet nagegaan; dat valt onder proef 4.

**Langflow.** `LANGFLOW_MCP_SERVER_ENABLED` staat standaard aan. Met `false` start Langflow geen MCP-servers. `LANGFLOW_MCP_SSE_ENABLED=false` alleen is niet genoeg.

Externe bronnen (geraadpleegd 29-10-2026):

- [open-notebook-mcp, README](https://github.com/Epochal-dev/open-notebook-mcp)
- [open-notebook-mcp, `server.py`](https://raw.githubusercontent.com/Epochal-dev/open-notebook-mcp/main/src/open_notebook_mcp/server.py)
- [Langflow als MCP-server](https://docs.langflow.org/mcp-server)
- [Langflow, globale variabelen](https://docs.langflow.org/configuration-global-variables)
- [Docker Compose-netwerken](https://docs.docker.com/compose/how-tos/networking/)

## Opties

Het FO maakt de keuzes. De opties gaan over de uitvoering van het filter:

1. **Filter per route in de router.** Elke router roept zelf een filterfunctie aan. Er zijn 42 routers; één vergeten aanroep lekt.
2. **Toegang per route met standaard weigeren, en het filter als verplichte parameter in de repositorylaag.** Een route zonder expliciete rechtendeclaratie start niet. Een leesmethode zonder classificatiebereik compileert niet (verplicht argument zonder standaardwaarde). Gekozen.
3. **Filter alleen in de database, met SurrealDB-rechten per record.** Sterk als tweede laag, maar het FO plaatst het filter in de API. Dit kan later als extra laag; het vervangt het API-filter niet.

## Voorstel

Optie 2, in zes onderdelen.

**A. Sleutels en rechten**

- `PasswordAuthMiddleware` en `OPEN_NOTEBOOK_PASSWORD` vervallen. In de plaats komt één sleutelmiddleware voor alle routes onder `/api`. Die weigert altijd als er geen geldige sleutel is.
- De sleutel komt in `Authorization: Bearer <sleutel>`. Dat sturen open-notebook-mcp en de interface al (`frontend/src/lib/api/client.ts:32`). `X-API-Key` vervalt; er is één manier.
- Hergebruik: tabel `agent_keys` met SHA-256-hash (`migrations/77.surrealql`), `AgentKeyService.authenticate` en de rechtencontrole uit `agent_auth.py`, die `require_right(...)` wordt. `AgentAuditMiddleware` gaat voor alle routes gelden in plaats van alleen voor `/api/v1/agents`.
- Een nieuwe migratie wijzigt het veld `permission` in `ASSERT $value IN ['lezen_extern', 'lezen_intern', 'beheren']`. Bestaande sleutels worden ingetrokken (`revoked = true`), omdat `read`/`write`/`admin` niets over classificatie zeggen. `_down` zet de oude waarden terug, maar laat de sleutels ingetrokken.
- De eerste sleutel met het recht beheren maakt een script vanaf de opdrachtregel. Het schrijft die sleutel als Docker secret weg en zet hem niet in een log. Meer sleutels maakt de beheerder via `/api/agent-keys` (recht beheren).
- Uitgezonderd van de sleutelplicht is alleen `/health`, dat geen inhoud teruggeeft. De uitzonderingen voor `/api/config`, `/docs`, `/openapi.json`, `/redoc`, `/api/v1/agents/` en de PDF-routes vervallen. `/api/auth/status` blijft, omdat het alleen meldt dat een sleutel nodig is.

**B. Rechten per route, standaard weigeren**

- Elke route declareert het minimale recht via een dependency. Een test loopt alle `application.routes` af en faalt bij een route zonder declaratie.
- `lezen_extern`, de sleutel van open-notebook-mcp, krijgt alleen:
  - `GET /api/notebooks` en `GET /api/notebooks/{id}`;
  - `GET /api/sources` en `GET /api/sources/{id}`;
  - `GET /api/notes` en `GET /api/notes/{id}`;
  - `POST /api/search`;
  - later de eindpunten voor herkomst en versiegeschiedenis uit FO 10.
- `lezen_intern` krijgt dezelfde leesroutes zonder het filter op extern, plus de exportgegevens voor Obsidian (ADR 0018).
- `beheren` krijgt de rest. Dat omvat schrijven, import, herclassificatie, sleutels, modellen, instellingen, triage, chat, ask, podcasts en transformaties.
- Chat en ask blijven bestaan (FO 13.2 punt 7), maar zijn voor open-notebook-mcp afgesloten. De tools `ask_question`, `execute_chat`, `create_*`, `update_*`, `delete_*`, `create_model` en `update_settings` krijgen HTTP 403. Er is geen eigen chatfunctie (FO 10).
- Routes voor entiteiten en de kennisgraaf staan niet op de lijst voor `lezen_extern`. Wanneer een entiteit extern zichtbaar mag zijn, wordt vastgelegd bij de herkomsteindpunten uit FO 10.

**C. Classificatiefilter in de repositorylaag**

- Het recht van de sleutel bepaalt een `ClassificationScope`: `lezen_extern` ziet alleen `extern`, `lezen_intern` en `beheren` zien alles.
- De leesmethoden van `SearchRepository`, `SourceRepository`, `NoteRepository` en `NotebookRepository` in `packages/surrealdb-service` krijgen `scope` als verplicht argument zonder standaardwaarde.
- `fn::vector_search` en `fn::text_search` (laatste definities in `migrations/9.surrealql` en `4.surrealql`) krijgen via een nieuwe migratie een parameter `$classifications`. Ze filteren in de query op `chunk.classification`, dus voordat een client of model iets ziet (FO 9, "externe uitvraag").
- Een gefilterd record geeft HTTP 404, niet 403, zodat het bestaan van een interne bron niet uitlekt.
- De huidige `fn::vector_search` rekent de cosinusgelijkenis uit in de `WHERE`-clausule (`migrations/9.surrealql:14-15`). Het classificatiefilter staat in dezelfde clausule en werkt dus vóór de rangschikking. Gebruikt een zoekroute een index met dichtstbijzijnde buren, dan kan een filter dat ná die index komt minder treffers geven. Dat raakt de volledigheid, niet harde eis 1. De bouwtaak meet het.

**D. MCP-servers**

- Het pakket `open-notebook-mcp` komt in een eigen container. De versie ligt vast op een uitgebrachte versie of commit. De container draait met `MCP_TRANSPORT=streamable-http` en `OPEN_NOTEBOOK_URL=http://open-notebook:5055`. `OPEN_NOTEBOOK_PASSWORD` bevat een sleutel met `lezen_extern`. Die komt uit een Docker secret: een klein startscript leest het geheimbestand, omdat het pakket alleen een omgevingsvariabele leest.
- De HTTP-transport van open-notebook-mcp heeft zelf geen authenticatie. Daarom luistert de poort alleen op `127.0.0.1` en via Tailscale. Wie de poort bereikt, ziet hooguit externe inhoud.
- Verwijderd, met script-ingang en tests:
  - `packages/surrealdb-service/src/surrealdb_service/mcp/` (`packages/surrealdb-service/pyproject.toml:30`);
  - `packages/llm-manager/src/llm_manager/mcp/` (`packages/llm-manager/pyproject.toml:38`);
  - `packages/file-manager/src/file_manager/mcp/` (`packages/file-manager/pyproject.toml:29`).
  
  `ARCHITECTURE.md` paragraaf 10 wordt bijgewerkt.
- Hergebruik: de querylogica van de tools `cite`, `related`, `get_node` en `get_entity_graph` uit `surrealdb-mcp` wordt de basis voor de herkomst- en navigatie-eindpunten die FO 10 in de API vraagt, mét classificatiebereik. Zo nodig krijgt open-notebook-mcp er tools bij.
- In Langflow staan `LANGFLOW_MCP_SERVER_ENABLED=false` en `LANGFLOW_ADD_PROJECTS_TO_MCP_SERVERS=false`.

**E. Extractiedienst en doorgeefroutes**

- De router `services_proxy` (`/api/services/*`, `services_proxy.py`) vervalt.
- De extractiedienst krijgt geen gepubliceerde poort meer en vervalt zelf met ADR 0012. PC.7 is daarmee niet meer nodig.

**F. Netwerk, poorten en geheimen in Compose**

- Netwerk `verwerking` met `internal: true` voor `surrealdb`, `open-notebook`, `llama-swap`, `docling-serve`, `whisperx`, `kev` en `open-notebook-mcp`.
- Netwerk `uitgaand` voor `langflow` (bronnen ophalen) en `claude-cli`.
- Een dienst die alleen aan een netwerk met `internal: true` hangt, kan geen poort naar de host publiceren. Dat staat in de Compose-documentatie en is in de bouwtaak te bevestigen. Daarom hangen `langflow` en `open-notebook-mcp` ook aan een netwerk `toegang`, en publiceren ze alleen op `127.0.0.1:<poort>`. Geen andere dienst publiceert een poort. Daarmee vervallen de gepubliceerde poorten van SurrealDB, de API, de interface en alle hulpdiensten.
- Geheimen staan in een `secrets:`-blok met bestanden buiten Git (bijvoorbeeld `secrets/` in `.gitignore`): de SurrealDB-inloggegevens, de sleutels van open-notebook-mcp en Langflow, het Hugging Face-token, de Claude-API-sleutel en de CLI-login. De diensten lezen ze via `/run/secrets/...`, eventueel met een startscript. `root`/`root` en `extra_hosts: host.docker.internal` vervallen.
- **Versleutelde opslag van cloudsleutels.** Na ADR 0006 doet de fork geen cloudaanroepen meer; daar vervallen de cloudsleutels, zoals `NVIDIA_API_KEY` en `api_key_env_var` voor cloudaanbieders. De Claude-API-sleutel staat alleen in Langflow, als globale variabele van het type Credential. Langflow versleutelt die in zijn database. Het geheim komt binnen via `LANGFLOW_VARIABLES_TO_GET_FROM_ENVIRONMENT`, gevuld vanuit een Docker secret.
- Images en versies liggen vast. `surrealdb/surrealdb:v2` met `pull_policy: always` (`docker-compose.yml:8`, `:25`) wordt een vaste versie.

## Gevolgen

- **Code:** in `apps/app-main/src/app_main/api/` worden `auth.py` en `agent_auth.py` herschreven; `app.py` krijgt de middleware zonder uitzonderingen en de routecontrole. De repositorylaag krijgt een verplicht `scope`. Er komen migraties voor `agent_keys.permission` en de zoekfuncties. Drie MCP-pakketonderdelen en `services_proxy` worden verwijderd.
- **Compose:** de nieuwe `docker-compose.signaalsysteem.yml` (stap 4) volgt onderdeel F en vervangt `docker-compose.yml`. Er blijven geen twee Compose-bestanden naast elkaar bestaan.
- **Tests (stopcondities):**
  - een routetest faalt bij een route zonder rechtendeclaratie;
  - een contracttest vult interne en externe testgegevens uit openbare bronnen en roept elke route voor `lezen_extern` aan; geen intern id of interne tekst mag in een antwoord staan;
  - dezelfde test via een echte open-notebook-mcp-client is proef 4;
  - een test dat elk verzoek zonder sleutel HTTP 401 krijgt.
- **Interface:** de interface moet inloggen met een sleutel in plaats van een wachtwoord. Het invoerveld blijft gelijk, het Bearer-schema ook. De PDF-weergave moet de header meesturen (ADR 0016).
- **Open punt:** wanneer een entiteit voor `lezen_extern` zichtbaar is. Dat wordt vastgelegd zodra de herkomsteindpunten uit FO 10 worden gebouwd.
