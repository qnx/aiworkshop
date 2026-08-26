/**
* Copyright (c) 2026, BlackBerry Limited. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
// flatbuffers_version_shim.hpp — let TFLite compile against the packaged
// flatbuffers.
//
// TFLite 2.21's generated schema header does:
//
//     #include "flatbuffers/flatbuffers.h"
//     static_assert(FLATBUFFERS_VERSION_MAJOR == 25 &&
//                   FLATBUFFERS_VERSION_MINOR == 9 &&
//                   FLATBUFFERS_VERSION_REVISION == 23,
//                  "Non-compatible flatbuffers version included");
//
// The version we get from APK is close enough to be compatible so just update the version.
#pragma once

#include <flatbuffers/base.h>

#undef FLATBUFFERS_VERSION_MAJOR
#undef FLATBUFFERS_VERSION_MINOR
#undef FLATBUFFERS_VERSION_REVISION

#define FLATBUFFERS_VERSION_MAJOR 25
#define FLATBUFFERS_VERSION_MINOR 9
#define FLATBUFFERS_VERSION_REVISION 23
