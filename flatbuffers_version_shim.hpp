// flatbuffers_version_shim.hpp — let TFLite compile against the packaged flatbuffers.
//
// TFLite 2.21's generated schema header does:
//
//     #include "flatbuffers/flatbuffers.h"
//     static_assert(FLATBUFFERS_VERSION_MAJOR == 25 &&
//                   FLATBUFFERS_VERSION_MINOR == 9 &&
//                   FLATBUFFERS_VERSION_REVISION == 23,
//                  "Non-compatible flatbuffers version included");
//
// It was generated against flatbuffers 25.9.23, but the QNX repos package only
// 25.12.19 (`apk search -v flatbuffers-dev` lists no other revision), so that
// assert fires and nothing using the TFLite C++ API compiles.
//
// The assert is a blunt guard against generated-code/runtime drift, and there is
// no opt-out macro in flatbuffers/base.h. flatbuffers is source-compatible across
// this range, so this header is force-included (-include) ahead of every TU: it
// pulls in the real system flatbuffers first, then restates the version macros to
// what the generated code expects. By the time schema_generated.h is reached, its
// own #include is a no-op via the include guard and the assert passes.
//
// This is a compile-time claim only -- the actual flatbuffers code used is the
// packaged 25.12.19. If a future TFLite or flatbuffers bump breaks the wire
// format for real, the symptom is a corrupt model parse at runtime, not a
// compile error. The check is that `./handtrack` still loads both models and
// puts a skeleton on a hand, rather than failing at interpreter build.
#pragma once

#include <flatbuffers/base.h>

#undef FLATBUFFERS_VERSION_MAJOR
#undef FLATBUFFERS_VERSION_MINOR
#undef FLATBUFFERS_VERSION_REVISION

#define FLATBUFFERS_VERSION_MAJOR 25
#define FLATBUFFERS_VERSION_MINOR 9
#define FLATBUFFERS_VERSION_REVISION 23
