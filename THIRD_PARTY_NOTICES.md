# Third-party notices

Recoleva is GPL-3.0-or-later. Runtime dependencies are installed separately and
retain their licenses; no third-party dependency code is bundled in this SDK
wheel. This is not a claim that the dependency stack is permissively licensed.

| Dependency | Observed license |
| --- | --- |
| Unidecode | GPL-2.0-or-later |
| SQLAlchemy, pgvector, pydantic, pydantic-core, annotated-types, typing-inspection | MIT |
| asyncpg | Apache-2.0 |
| greenlet | MIT AND PSF-2.0 |
| typing-extensions | PSF-2.0 |

Unidecode's transliteration behavior remains unchanged; its tables and code
are supplied by its separately installed distribution. The upstream project is
https://github.com/avian2/unidecode. Retain its own copyright/license notices
when redistributing it. Likewise, distributors that bundle other dependencies
must include the corresponding upstream notices and satisfy their licenses.

DEPENDENCY_LICENSES.json records bootstrap versions and observed metadata.
The release's `release.json` records installed versions and license metadata;
that is evidence of the tested environment, not a license grant from this SDK
for third-party material. Build/test tools are separate development dependencies.

The GPL text in LICENSE is the SPDX license-list-data GPL-3.0-or-later text,
which reproduces the GNU General Public License version 3. The initial SDK
source/test provenance is recorded in EXPORT_PROVENANCE.json and
BOOTSTRAP_PROVENANCE.json; no private Git history or conversation data is included.
