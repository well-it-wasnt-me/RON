# CHANGELOG

<!-- version list -->

## v2.0.1 (2026-08-18)

### Bug Fixes

- **telegram**: Small telegram fix in is command parsing
  ([`c6dfadd`](https://github.com/well-it-wasnt-me/RON/commit/c6dfaddfee349b85f9d9933657e21a7e282f0a89))

- **web**: Calibration and more
  ([`1bd92a9`](https://github.com/well-it-wasnt-me/RON/commit/1bd92a97edffc02f3fade9904ca83b8fe82ede96))

### Chores

- **camera**: Re-instating rtsp camera bridge
  ([`ef07354`](https://github.com/well-it-wasnt-me/RON/commit/ef0735411e1f23d4ca4eece60910396652c4fcf6))


## v2.0.0 (2026-08-18)

### Bug Fixes

- Ai slop...every. single. time.
  ([`29884db`](https://github.com/well-it-wasnt-me/RON/commit/29884db21710f403bb21a301f66b1cf2e120c2e4))

- **learning**: Correct gradient normalization, recorder state ordering, and thread safety
  ([`2513432`](https://github.com/well-it-wasnt-me/RON/commit/2513432d00356f0e00b1e7c357c56c167e3aba0b))

- **learning**: Correct gradient normalization, recorder state ordering, and thread safety
  ([`700d7c1`](https://github.com/well-it-wasnt-me/RON/commit/700d7c1e327b82bd925ca8b7a5cf7bfffff0f812))

### Chores

- Delete useless stuff [skip ci]
  ([`679d41d`](https://github.com/well-it-wasnt-me/RON/commit/679d41d83073f45c930091ad59387879ad44186c))

- Git ignore update
  ([`db219f6`](https://github.com/well-it-wasnt-me/RON/commit/db219f63e51fde966e7c2cef4d12a8c1b224fa32))

- Readme and pyproject update
  ([`0eb6391`](https://github.com/well-it-wasnt-me/RON/commit/0eb63917c64bb046fdf2639f57fff8dc65e7ee56))

- Small linting fixes
  ([`4fc288b`](https://github.com/well-it-wasnt-me/RON/commit/4fc288bca185cf8ff236f11e2be0cc61bcbef1c4))

### Continuous Integration

- Added semversioning, better validation and reorganized steps
  ([`eeb6465`](https://github.com/well-it-wasnt-me/RON/commit/eeb6465f14126c380367a788cedcf537c5da8a21))

- Github page automatic build
  ([`f9551da`](https://github.com/well-it-wasnt-me/RON/commit/f9551da5ca295c2d0d96f9d91a3de1894a010bd5))

### Documentation

- Forgot to add telegram doc to mkdocs [skip ci]
  ([`9fe90f9`](https://github.com/well-it-wasnt-me/RON/commit/9fe90f95f1b116c54daf7e998e6feb735794ead0))

- Major documentation update and code lint
  ([`93f7944`](https://github.com/well-it-wasnt-me/RON/commit/93f794411e050dd43107351161cd0c20291a66bf))

- Major documentation update and code lint
  ([`f2e737d`](https://github.com/well-it-wasnt-me/RON/commit/f2e737dd9e57d76f9718d7c8fe3695fbaf60a6f2))

### Features

- Telegram bridge
  ([`7d577e0`](https://github.com/well-it-wasnt-me/RON/commit/7d577e09c4f410ad703568b65409be7d5163f21e))

- **learing**: Multimodal
  ([`fe5cb66`](https://github.com/well-it-wasnt-me/RON/commit/fe5cb664382cc46170fd12a595fccab9ff418723))

- **learing**: Multimodal [skip ci]
  ([`ed73134`](https://github.com/well-it-wasnt-me/RON/commit/ed731341cc67df6ad06bf65d22e29a1c2e9b17ff))

### Breaking Changes

- **learing**: The ExperienceRecorder no longer creates experiences from observation events. Use
  begin_transition(action_index) / complete_transition(pending, reward) instead. The legacy record()
  method still works for manual/pre-collected data.


## v1.0.0 (2026-08-15)

- Initial Release
