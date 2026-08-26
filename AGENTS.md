<!-- bmad:context -->
<!-- Verified 2026-08-11 against d3ce528 (плюс незакомічені зміни в робочому дереві). Managed by bmad-project-context; edits inside this block are replaced on refresh. Keep anything you want preserved outside the markers. -->

## veda

Каркас рушія інференсу LLM на C++23, CMake ≥ 3.20. Наразі `Facade` лише друкує імена чотирьох
заготовок (`Tensor`, `WeightsLoader`, `BytePairEncoding`, `Sampler`) — реалізації немає, її
доведеться писати з нуля. Планувальні артефакти BMad лежать у `_bmad-output/`.

## Running and verifying

- Тестів немає: `enable_testing()` не викликано, тестових таргетів нема. Єдина перевірка — запустити
  бінарник і подивитися вивід.
- Для clangd генеруй свій індекс: `cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`, і вкажи
  clangd на `build/`.

## Conventions that differ from defaults

- Підключай заголовки плоско — `#include "Tensor.h"`, не `#include "libs/tensor/Tensor.h"`. Кожен
  каталог бібліотеки окремо перелічений у `target_include_directories`.
- Додаючи `src/libs/<name>/`, зроби обидві правки в `CMakeLists.txt`: впиши і `.cpp`, і `.h` у
  `add_executable(veda ...)`, і додай каталог у `target_include_directories`. `add_subdirectory` та
  окремих бібліотечних таргетів у проєкті немає.

## Known pitfalls

- Не складай у `cmake-build-debug/` і `cmake-build-debug-docker/` — їхній `CMAKE_HOME_DIRECTORY`
  вказує на `/Users/konstantinohotnik/apps/veda` і `/tmp/veda`. Обидві теки в `.gitignore`, у git їх
  немає. Складай у `build/`.
- `docker-compose.yml` монтує `.:/app` поверх `/app`, ховаючи `build/` з образу — `./build/veda`
  всередині контейнера не існує, поки не складеш там заново.

<!-- /bmad:context -->
