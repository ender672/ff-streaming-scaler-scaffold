/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef MOZILLA_GFX_STREAMINGSCALERINTERNAL_H_
#define MOZILLA_GFX_STREAMINGSCALERINTERNAL_H_

#include <array>

namespace mozilla::gfx {

/* Lookup table mapping byte values [0,255] to float values [0.0, 1.0]. */
inline constexpr auto gI2fMap = [] {
  std::array<float, 256> map{};
  for (int i = 0; i < 256; i++) {
    map[i] = float(i) / 255.0f;
  }
  return map;
}();

}  // namespace mozilla::gfx

#endif  // MOZILLA_GFX_STREAMINGSCALERINTERNAL_H_
