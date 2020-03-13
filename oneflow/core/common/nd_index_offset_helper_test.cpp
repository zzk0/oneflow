#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace test {

TEST(NdIndexOffsetHelper, test) {
  const int64_t d0_max = 3;
  const int64_t d1_max = 4;
  const int64_t d2_max = 5;
  const NdIndexOffsetHelper<int64_t, 3> helper(d0_max, d1_max, d2_max);
  for (int64_t d0 = 0; d0 < d0_max; ++d0) {
    const int64_t offset0 = d0 * d1_max * d2_max;
    std::vector<int64_t> expected0({d0});
    std::vector<int64_t> dims0;
    dims0.resize(1);
    helper.OffsetToNdIndex(offset0, &dims0.at(0));
    ASSERT_EQ(expected0, dims0);
    helper.OffsetToNdIndex(offset0, dims0.data());
    ASSERT_EQ(expected0, dims0);
    helper.OffsetToNdIndex(offset0, 1, dims0.data());
    ASSERT_EQ(expected0, dims0);
    ASSERT_EQ(offset0, helper.NdIndexToOffset(d0));
    ASSERT_EQ(offset0, helper.NdIndexToOffset(1, expected0.data()));

    for (int64_t d1 = 0; d1 < d1_max; ++d1) {
      const int64_t offset1 = offset0 + d1 * d2_max;
      std::vector<int64_t> expected1({d0, d1});
      std::vector<int64_t> dims1;
      dims1.resize(2);
      helper.OffsetToNdIndex(offset1, &dims1.at(0), &dims1.at(1));
      ASSERT_EQ(expected1, dims1);
      helper.OffsetToNdIndex(offset1, dims1.data());
      ASSERT_EQ(expected1, dims1);
      helper.OffsetToNdIndex(offset1, 2, dims1.data());
      ASSERT_EQ(expected1, dims1);
      ASSERT_EQ(offset1, helper.NdIndexToOffset(d0, d1));
      ASSERT_EQ(offset1, helper.NdIndexToOffset(2, expected1.data()));

      for (int64_t d2 = 0; d2 < d2_max; ++d2) {
        const int64_t offset2 = offset1 + d2;
        std::vector<int64_t> expected2({d0, d1, d2});
        std::vector<int64_t> dims2;
        dims2.resize(3);
        helper.OffsetToNdIndex(offset2, &dims2.at(0), &dims2.at(1), &dims2.at(2));
        ASSERT_EQ(expected2, dims2);
        helper.OffsetToNdIndex(offset2, dims2.data());
        ASSERT_EQ(expected2, dims2);
        helper.OffsetToNdIndex(offset2, 3, dims2.data());
        ASSERT_EQ(expected2, dims2);
        ASSERT_EQ(offset2, helper.NdIndexToOffset(d0, d1, d2));
        ASSERT_EQ(offset2, helper.NdIndexToOffset(expected2.data()));
        ASSERT_EQ(offset2, helper.NdIndexToOffset(3, expected2.data()));
      }
    }
  }
}

}  // namespace test

}  // namespace oneflow