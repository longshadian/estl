#pragma once

#include <cstdint>
#include <cstddef>

namespace replaylib {

void encrypt(const uint8_t* src, size_t len, const uint8_t* key, size_t key_len, uint8_t* out);
void decrypt(const uint8_t* src, size_t len, const uint8_t* key, size_t key_len, uint8_t* out);

} // replaylib
