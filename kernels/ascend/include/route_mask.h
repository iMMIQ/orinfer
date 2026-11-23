namespace tl::ascend {
template <uint32_t Width, uint32_t Selected>
__aicore__ inline void orinfer_compact_routes(
    AscendC::LocalTensor<int32_t> destination,
    AscendC::LocalTensor<int32_t> positions,
    AscendC::LocalTensor<uint8_t> masks, bool, uint32_t live) {
  static_assert(Width == 256 && Selected >= 1 && Selected <= 8);
  constexpr uint16_t lane_mask = (1u << Selected) - 1;
  auto selected = masks[Width / 8].ReinterpretCast<uint16_t>();
  auto experts = masks.ReinterpretCast<uint16_t>();
  AscendC::Duplicate(selected, static_cast<uint16_t>(lane_mask * 257), Width / 16);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::And(experts, experts, selected, Width / 16);
  AscendC::PipeBarrier<PIPE_V>();
  uint64_t count = 0;
  AscendC::GatherMask(destination, positions, masks.ReinterpretCast<uint32_t>(),
                      true, live, {1, 1, 8, 0}, count);
  AscendC::SetFlag<AscendC::HardEvent::V_S>(5);
  AscendC::WaitFlag<AscendC::HardEvent::V_S>(5);
  destination.SetValue(Width, static_cast<int32_t>(count));
}
}
