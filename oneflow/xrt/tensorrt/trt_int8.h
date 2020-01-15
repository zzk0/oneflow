#ifndef ONEFLOW_XRT_TENSORRT_TRT_INT8_H
#define ONEFLOE_XRT_TENSORRT_TRT_INT8_H


//#include "oneflow/xrt/tensorrt/trt_builder.h"
#include "NvInfer.h"
#include "glog/logging.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <assert.h>
#include <numeric>
#include <algorithm>
#include <iterator>



namespace oneflow {
namespace xrt {
namespace tensorrt {

enum class XrtInt8CalibrationAlgoType : int {
  kLegacyCalibration = 0,
  kEntropyCalibration = 1,
  kEntropyCalibration_2 = 2, // this is the calibrator for DLA
  kMinMaxCalibration = 3, // this is the calibrator for NLP tasks
};

template <int C, int H, int W>
struct JPEG
{
    std::string magic, fileName;
    int h, w, max;
    uint8_t buffer[C * H * W];
};

template <int C, int H, int W>
inline void readJPEGFile(const std::string& filename, JPEG<C, H, W>& jpeg)
{
    jpeg.fileName = filename;
    std::ifstream infile(filename, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> jpeg.magic >> jpeg.w >> jpeg.h >> jpeg.max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(jpeg.buffer), jpeg.w * jpeg.h * 3);
}

// Swaps endianness of an integral type.
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline T swapEndianness(const T& value)
{
    uint8_t bytes[sizeof(T)];
    for (int i = 0; i < static_cast<int>(sizeof(T)); ++i)
    {
        bytes[sizeof(T) - 1 - i] = *(reinterpret_cast<const uint8_t*>(&value) + i);
    }
    return *reinterpret_cast<T*>(bytes);
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

// Locate path to file, given its filename or filepath suffix and possible dirs it might lie in
// Function will also walk back MAX_DEPTH dirs from CWD to check for such a file path
inline std::string locateFile(const std::string& filepathSuffix, const std::vector<std::string>& directories)
{
    const int MAX_DEPTH{10};
    bool found{false};
    std::string filepath;

    for (auto& dir : directories)
    {
        if (!dir.empty() && dir.back() != '/')
        {
#ifdef _MSC_VER
            filepath = dir + "\\" + filepathSuffix;
#else
            filepath = dir + "/" + filepathSuffix;
#endif
        }
        else
            filepath = dir + filepathSuffix;

        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            std::ifstream checkFile(filepath);
            found = checkFile.is_open();
            if (found)
                break;
            filepath = "../" + filepath; // Try again in parent dir
        }

        if (found)
        {
            break;
        }

        filepath.clear();
    }

    if (filepath.empty())
    {
        std::string directoryList = std::accumulate(directories.begin() + 1, directories.end(), directories.front(),
            [](const std::string& a, const std::string& b) { return a + "\n\t" + b; });
        std::cout << "Could not find " << filepathSuffix << " in data directories:\n\t" << directoryList << std::endl;
        std::cout << "&&&& FAILED" << std::endl;
        exit(EXIT_FAILURE);
    }
    return filepath;
}

class TrtBatchStream
{

public:
/*  TrtBatchStream(int batch_size, int max_batches, std::vector<int> images_dims, const std::string prefix, const std::string suffix,
      const std::vector<std::string> dirs)
    : mBatchSize{batch_size}
  , mMaxBatches{max_batches}
  , mDims{images_dims}
  , mPrefix(prefix)
    , mSuffix(suffix)
    , mDataDir(dirs)

    {
      file_name = mPrefix + std::string("0") + mSuffix, mDataDir;
      std::cout << "file_name: " << file_name << std::endl;
      FILE* file = fopen(locateFile(file_name).c_str(), "rb");
      assert(file != nullptr);
    //  int d[4]=[3, 3, 224, 224];
        size_t readSize = fread(, sizeof(int), 4, file);
        assert(readSize == 4);
        mDims.nbDims = 4;  // The number of dimensions.
        //mDims.d[0] = images_dims[0]; // Batch Size
        //mDims.d[1] = images_dims[1]; // Channels
        //mDims.d[2] = images_dims[2]; // Height
        //mDims.d[3] = images_dims[3]; // Width
        assert(mDims.d[0] > 0 && mDims.d[1] > 0 && mDims.d[2] > 0 && mDims.d[3] > 0);
        fclose(file);

        mImageSize = mDims.d[1] * mDims.d[2] * mDims.d[3];
        mBatch.resize(mBatchSize * mImageSize, 0);
        mLabels.resize(mBatchSize, 0);
    //! \class EntropyCalibratorImpl
//!
//! \brief Implements common functionality for Entropy calibrators.
//!
template <typename TBatchStream>
class EntropyCalibratorImpl
{
public:
    EntropyCalibratorImpl(
        TBatchStream stream, int firstBatch, std::string networkName, const char* inputBlobName, bool readCache = true)
        : mStream{stream}
        , mCalibrationTableName("CalibrationTable" + networkName)
        , mInputBlobName(inputBlobName)
        , mReadCache(readCache)
    {
        nvinfer1::Dims dims = mStream.getDims();
        mInputCount = volume(dims) * mStream.getBatchSize();
        CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
        mStream.reset(firstBatch);
    }

    virtual ~EntropyCalibratorImpl()
    {
        CHECK(cudaFree(mDeviceInput));
    }

    int getBatchSize() const
    {
        return mStream.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings)
    {
        if (!mStream.next())
        {
            return false;
        }
        CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        assert(!strcmp(names[0], mInputBlobName));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length)
    {
        mCalibrationCache.clear();
        std::ifstream input(mCalibrationTableName, std::ios::binary);
        input >> std::noskipws;
        if (mReadCache && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                std::back_inserter(mCalibrationCache));
        }
        length = mCalibrationCache.size();
        return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length)
    {
        std::ofstream output(mCalibrationTableName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    TBatchStream mStream;
    size_t mInputCount;
    std::string mCalibrationTableName;
    const char* mInputBlobName;
    bool mReadCache{true};
    void* mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache;
};

//! \class Int8EntropyCalibrator2
//!
//! \brief Implements Entropy calibrator 2.
//!  CalibrationAlgoType is kENTROPY_CALIBRATION_2.
//!
template <typename TBatchStream>
class Int8EntropyCalibrator2 : public IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator2(
        TBatchStream stream, int firstBatch, const char* networkName, const char* inputBlobName, bool readCache = true)
        : mImpl(stream, firstBatch, networkName, inputBlobName, readCache)
    {
    }

    int getBatchSize() const 
    {
        return mImpl.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) 
    {
        return mImpl.getBatch(bindings, names, nbBindings);
    }

    const void* readCalibrationCache(size_t& length) 
    {
        return mImpl.readCalibrationCache(length);
    }

    void writeCalibrationCache(const void* cache, size_t length) 
    {
        mImpl.writeCalibrationCache(cache, length);
    }

private:
    EntropyCalibratorImpl<TBatchStream> mImpl;
};    mFileBatch.resize(mDims.d[0] * mImageSize, 0);
        mFileLabels.resize(mDims.d[0], 0);
        reset(0);
    }
*/
    TrtBatchStream(
        int batch_size, int max_batches, nvinfer1::Dims dims, std::string listFile, std::vector<std::string> dirs)
        : mBatchSize(batch_size)
        , mMaxBatches(max_batches)
        , mDims(dims)
        , mListFile(listFile)
        , mDataDir(dirs)
    {
        mImageSize = mDims.d[1] * mDims.d[2] * mDims.d[3];
        mBatch.resize(mBatchSize * mImageSize, 0);
        mLabels.resize(mBatchSize, 0);
        mFileBatch.resize(mDims.d[0] * mImageSize, 0);
        mFileLabels.resize(mDims.d[0], 0);
        reset(0);
    }

    // Resets data members
    void reset(int firstBatch)
    {
        mBatchCount = 0;
        mFileCount = 0;
        mFileBatchPos = mDims.d[0];
        skip(firstBatch);
    }

    // Advance to next batch and return true, or return false if there is no batch left.
    bool next()
    {
        if (mBatchCount == mMaxBatches)
        {
            return false;
        }

        for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize)
        {
            assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.d[0]);
            if (mFileBatchPos == mDims.d[0] && !update())
            {
                return false;
            }

            // copy the smaller of: elements left to fulfill the request, or elements left in the file buffer.
            csize = std::min(mBatchSize - batchPos, mDims.d[0] - mFileBatchPos);
            std::copy_n(
                getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
            std::copy_n(getFileLabels() + mFileBatchPos, csize, getLabels() + batchPos);
        }
        mBatchCount++;
        return true;
    }

    // Skips the batches
    void skip(int skipCount) 
    {
        if (mBatchSize >= mDims.d[0] && mBatchSize % mDims.d[0] == 0 && mFileBatchPos == mDims.d[0])
        {
            mFileCount += skipCount * mBatchSize / mDims.d[0];
            return;
        }

        int x = mBatchCount;
        for (int i = 0; i < skipCount; i++)
        {
            next();
        }
        mBatchCount = x;
    }

    float* getBatch() 
    {
        return mBatch.data();
    }

    float* getLabels() 
    {
        return mLabels.data();
    }

    int getBatchesRead() const 
    {
        return mBatchCount;
    }

    int getBatchSize() const 
    {
        return mBatchSize;
    }

    nvinfer1::Dims getDims() const 
    {
        return mDims;
    }

private:
    float* getFileBatch()
    {
        return mFileBatch.data();
    }

    float* getFileLabels()
    {
        return mFileLabels.data();
    }

    bool update()
    {
        if (mListFile.empty())
        {
            std::string inputFileName = locateFile(mPrefix + std::to_string(mFileCount++) + mSuffix, mDataDir);
            FILE* file = fopen(inputFileName.c_str(), "rb");
            if (!file)
            {
                return false;
            }

            int d[4];
            size_t readSize = fread(d, sizeof(int), 4, file);
            assert(readSize == 4);
            assert(mDims.d[0] == d[0] && mDims.d[1] == d[1] && mDims.d[2] == d[2] && mDims.d[3] == d[3]);
            size_t readInputCount = fread(getFileBatch(), sizeof(float), mDims.d[0] * mImageSize, file);
            assert(readInputCount == size_t(mDims.d[0] * mImageSize));
            size_t readLabelCount = fread(getFileLabels(), sizeof(float), mDims.d[0], file);
            assert(readLabelCount == 0 || readLabelCount == size_t(mDims.d[0]));

            fclose(file);
        }
        else
        {
            std::vector<std::string> fNames;
            std::ifstream file(locateFile(mListFile, mDataDir), std::ios::binary);
            if (!file)
            {
                return false;
            }

            LOG(INFO) << "Batch #" << mFileCount << std::endl;
            file.seekg(((mBatchCount * mBatchSize)) * 7);

            for (int i = 1; i <= mBatchSize; i++)
            {
                std::string sName;
                std::getline(file, sName);
                sName = sName + ".JPEG";
                LOG(INFO) << "Calibrating with file " << sName << std::endl;
                fNames.emplace_back(sName);
            }

            mFileCount++;

            const int imageC = 3;
            const int imageH = 300;
            const int imageW = 300;
            std::vector<JPEG<imageC, imageH, imageW>> jpegs(fNames.size());
            for (uint32_t i = 0; i < fNames.size(); ++i)
            {
                readJPEGFile(locateFile(fNames[i], mDataDir), jpegs[i]);
            }

            std::vector<float> data(volume(mDims));
            const float scale = 2.0 / 255.0;
            const float bias = 1.0;
            long int volChl = mDims.d[2] * mDims.d[3];

            // Normalize input data
            for (int i = 0, volImg = mDims.d[1] * mDims.d[2] * mDims.d[3]; i < mBatchSize; ++i)
            {
                for (int c = 0; c < mDims.d[1]; ++c)
                {
                    for (int j = 0; j < volChl; ++j)
                    {
                        data[i * volImg + c * volChl + j] = scale * float(jpegs[i].buffer[j * mDims.d[1] + c]) - bias;
                    }
                }
            }

            std::copy_n(data.data(), mDims.d[0] * mImageSize, getFileBatch());
        }

        mFileBatchPos = 0;
        return true;
    }
    int mBatchSize{0};
    int mMaxBatches{0};
    int mBatchCount{0};
    int mFileCount{0};
    int mFileBatchPos{0};
    int mImageSize{0};
    std::vector<float> mBatch;         //!< Data for the batch
    std::vector<float> mLabels;        //!< Labels for the batch
    std::vector<float> mFileBatch;     //!< List of image files
    std::vector<float> mFileLabels;    //!< List of label files
    std::string mPrefix;               //!< Batch file name prefix
    std::string mSuffix;               //!< Batch file name suffix
    nvinfer1::Dims mDims;              //!< Input dimensions
    std::string mListFile;             //!< File name of the list of image names
    std::vector<std::string> mDataDir; //!< Directories where the files can be found
};

//! \class EntropyCalibratorImpl
//!
//! \brief Implements common functionality for Entropy calibrators.
//!
template <typename TBatchStream>
class EntropyCalibratorImpl
{
public:
    EntropyCalibratorImpl(
        TBatchStream stream, int firstBatch, const char* inputBlobName, bool readCache = true)
        : mStream{stream}
        , mCalibrationTableName("CalibrationTable")
        , mInputBlobName(inputBlobName)
        , mReadCache(readCache)
    {
        nvinfer1::Dims dims = mStream.getDims();
        mInputCount = volume(dims) * mStream.getBatchSize();
        CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
        mStream.reset(firstBatch);
    }

    virtual ~EntropyCalibratorImpl()
    {
        CHECK(cudaFree(mDeviceInput));
    }

    int getBatchSize() const
    {
        return mStream.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings)
    {
        if (!mStream.next())
        {
            return false;
        }
        CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        assert(!strcmp(names[0], mInputBlobName));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length)
    {
        mCalibrationCache.clear();
        std::ifstream input(mCalibrationTableName, std::ios::binary);
        input >> std::noskipws;
        if (mReadCache && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                std::back_inserter(mCalibrationCache));
        }
        length = mCalibrationCache.size();
        return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length)
    {
        std::ofstream output(mCalibrationTableName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    TBatchStream mStream;
    size_t mInputCount;
    std::string mCalibrationTableName;
    const char* mInputBlobName;
    bool mReadCache{true};
    void* mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache;
};

//! \class Int8EntropyCalibrator2
//!
//! \brief Implements Entropy calibrator 2.
//!  CalibrationAlgoType is kENTROPY_CALIBRATION_2.
//!
template <typename TBatchStream>
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator {
public:
    Int8EntropyCalibrator(
        TBatchStream stream, int firstBatch, const char* inputBlobName, bool readCache = true)
        : mImpl(stream, firstBatch, inputBlobName, readCache) {}

    int getBatchSize() const 
    {
        return mImpl.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) 
    {
        return mImpl.getBatch(bindings, names, nbBindings);
    }

    const void* readCalibrationCache(size_t& length) 
    {
        return mImpl.readCalibrationCache(length);
    }

    void writeCalibrationCache(const void* cache, size_t length) 
    {
        mImpl.writeCalibrationCache(cache, length);
    }

private:
    EntropyCalibratorImpl<TBatchStream> mImpl;
};


}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow

#endif  //ONEFLOW_XRT_TENSORRT_TRT_INT8_H
