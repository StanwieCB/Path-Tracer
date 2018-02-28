// Yiheng Zhang 515030910216
// CG Course Assignment2
// part of the code like math ultilities are taken from smallVCM open source code
#include <vector>
#include <cmath>
#include <random>
#include <time.h>
#include <cstdlib>
#include <string>
#include <set>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <fstream>

#define EPS_PHONG 1e-3f
#define PI_F     3.14159265358979f
#define INV_PI_F (1.f / PI_F)
#define EPS_COSINE 1e-6f
#define EPS_RAY    1e-4f

//////////////////////////////////////////////////////////////////////////
// Math and Utilities
// References: http://www.smallvcm.com
#pragma region MATHUTIL
template<typename T>
T Sqr(const T& a) { return a*a; }

typedef unsigned uint;
template<typename T>
class Vec2x
{
public:

	Vec2x() {}
	Vec2x(T a) :x(a), y(a) {}
	Vec2x(T a, T b) :x(a), y(b) {}

	const T& Get(int a) const { return reinterpret_cast<const T*>(this)[a]; }
	T&       Get(int a) { return reinterpret_cast<T*>(this)[a]; }

	// unary minus
	Vec2x<T> operator-() const
	{
		Vec2x<T> res; for (int i = 0; i < 2; i++) res.Get(i) = -Get(i); return res;
	}

	// binary operations
	friend Vec2x<T> operator+(const Vec2x& a, const Vec2x& b)
	{
		Vec2x<T> res; for (int i = 0; i < 2; i++) res.Get(i) = a.Get(i) + b.Get(i); return res;
	}
	friend Vec2x<T> operator-(const Vec2x& a, const Vec2x& b)
	{
		Vec2x<T> res; for (int i = 0; i < 2; i++) res.Get(i) = a.Get(i) - b.Get(i); return res;
	}
	friend Vec2x<T> operator*(const Vec2x& a, const Vec2x& b)
	{
		Vec2x<T> res; for (int i = 0; i < 2; i++) res.Get(i) = a.Get(i) * b.Get(i); return res;
	}
	friend Vec2x<T> operator/(const Vec2x& a, const Vec2x& b)
	{
		Vec2x<T> res; for (int i = 0; i < 2; i++) res.Get(i) = a.Get(i) / b.Get(i); return res;
	}

	Vec2x<T>& operator+=(const Vec2x& a)
	{
		for (int i = 0; i < 2; i++) Get(i) += a.Get(i); return *this;
	}
	Vec2x<T>& operator-=(const Vec2x& a)
	{
		for (int i = 0; i < 2; i++) Get(i) -= a.Get(i); return *this;
	}
	Vec2x<T>& operator*=(const Vec2x& a)
	{
		for (int i = 0; i < 2; i++) Get(i) *= a.Get(i); return *this;
	}
	Vec2x<T>& operator/=(const Vec2x& a)
	{
		for (int i = 0; i < 2; i++) Get(i) /= a.Get(i); return *this;
	}

	friend T Dot(const Vec2x& a, const Vec2x& b)
	{
		T res(0); for (int i = 0; i < 2; i++) res += a.Get(i) * b.Get(i); return res;
	}

public:

	T x, y;
};

typedef Vec2x<float> Vec2f;
typedef Vec2x<int>   Vec2i;

template<typename T>
class Vec3x
{
public:

	Vec3x() {}
	Vec3x(T a) :x(a), y(a), z(a) {}
	Vec3x(T a, T b, T c) :x(a), y(b), z(c) {}

	const T& Get(int a) const { return reinterpret_cast<const T*>(this)[a]; }
	T&       Get(int a) { return reinterpret_cast<T*>(this)[a]; }
	Vec2x<T> GetXY() const { return Vec2x<T>(x, y); }
	T        Max()   const { T res = Get(0); for (int i = 1; i < 3; i++) res = std::max(res, Get(i)); return res; }

	bool     IsZero() const
	{
		for (int i = 0; i < 3; i++)
			if (Get(i) != 0)
				return false;
		return true;
	}

	// unary minus
	Vec3x<T> operator-() const
	{
		Vec3x<T> res; for (int i = 0; i < 3; i++) res.Get(i) = -Get(i); return res;
	}

	// binary operations
	friend Vec3x<T> operator+(const Vec3x& a, const Vec3x& b)
	{
		Vec3x<T> res; for (int i = 0; i < 3; i++) res.Get(i) = a.Get(i) + b.Get(i); return res;
	}
	friend Vec3x<T> operator-(const Vec3x& a, const Vec3x& b)
	{
		Vec3x<T> res; for (int i = 0; i < 3; i++) res.Get(i) = a.Get(i) - b.Get(i); return res;
	}
	friend Vec3x<T> operator*(const Vec3x& a, const Vec3x& b)
	{
		Vec3x<T> res; for (int i = 0; i < 3; i++) res.Get(i) = a.Get(i) * b.Get(i); return res;
	}
	friend Vec3x<T> operator/(const Vec3x& a, const Vec3x& b)
	{
		Vec3x<T> res; for (int i = 0; i < 3; i++) res.Get(i) = a.Get(i) / b.Get(i); return res;
	}

	Vec3x<T>& operator+=(const Vec3x& a)
	{
		for (int i = 0; i < 3; i++) Get(i) += a.Get(i); return *this;
	}
	Vec3x<T>& operator-=(const Vec3x& a)
	{
		for (int i = 0; i < 3; i++) Get(i) -= a.Get(i); return *this;
	}
	Vec3x<T>& operator*=(const Vec3x& a)
	{
		for (int i = 0; i < 3; i++) Get(i) *= a.Get(i); return *this;
	}
	Vec3x<T>& operator/=(const Vec3x& a)
	{
		for (int i = 0; i < 3; i++) Get(i) /= a.Get(i); return *this;
	}

	friend T Dot(const Vec3x& a, const Vec3x& b)
	{
		T res(0); for (int i = 0; i < 3; i++) res += a.Get(i) * b.Get(i); return res;
	}

	float    LenSqr() const { return Dot(*this, *this); }
	float    Length() const { return std::sqrt(LenSqr()); }

public:

	T x, y, z;
};

typedef Vec3x<float> Vec3f;
typedef Vec3x<int>   Vec3i;

Vec3f Cross(
	const Vec3f &a,
	const Vec3f &b)
{
	Vec3f res;
	res.x = a.y * b.z - a.z * b.y;
	res.y = a.z * b.x - a.x * b.z;
	res.z = a.x * b.y - a.y * b.x;
	return res;
}

Vec3f Normalize(const Vec3f& a)
{
	const float lenSqr = Dot(a, a);
	const float len = std::sqrt(lenSqr);
	return a / len;
}

class Mat4f
{
public:

	Mat4f() {}
	Mat4f(float a) { for (int i = 0; i < 16; i++) GetPtr()[i] = a; }

	const float* GetPtr() const { return reinterpret_cast<const float*>(this); }
	float*       GetPtr() { return reinterpret_cast<float*>(this); }

	const float& Get(int r, int c) const { return GetPtr()[r + c * 4]; }
	float&       Get(int r, int c) { return GetPtr()[r + c * 4]; }

	void SetRow(int r, float a, float b, float c, float d)
	{
		Get(r, 0) = a;
		Get(r, 1) = b;
		Get(r, 2) = c;
		Get(r, 3) = d;
	}

	void SetRow(int r, const Vec3f &a, float b)
	{
		for (int i = 0; i < 3; i++)
			Get(r, i) = a.Get(i);

		Get(r, 3) = b;
	}

	Vec3f TransformVector(const Vec3f& aVec) const
	{
		Vec3f res(0);
		for (int r = 0; r < 3; r++)
			for (int c = 0; c < 3; c++)
				res.Get(r) += aVec.Get(c) * Get(r, c);

		return res;
	}

	Vec3f TransformPoint(const Vec3f& aVec) const
	{
		float w = Get(3, 3);

		for (int c = 0; c < 3; c++)
			w += Get(3, c) * aVec.Get(c);

		const float invW = 1.f / w;

		Vec3f res(0);

		for (int r = 0; r < 3; r++)
		{
			res.Get(r) = Get(r, 3);

			for (int c = 0; c < 3; c++)
				res.Get(r) += aVec.Get(c) * Get(r, c);

			res.Get(r) *= invW;
		}
		return res;
	}

	static Mat4f Zero() { Mat4f res(0); return res; }

	static Mat4f Identity()
	{
		Mat4f res(0);
		for (int i = 0; i < 4; i++) res.Get(i, i) = 1.f;
		return res;
	}

	static Mat4f Scale(const Vec3f& aScale)
	{
		Mat4f res = Mat4f::Identity();
		for (int i = 0; i < 3; i++) res.Get(i, i) = aScale.Get(i);
		res.Get(3, 3) = 1;
		return res;
	}

	static Mat4f Translate(const Vec3f& aScale)
	{
		Mat4f res = Mat4f::Identity();
		for (int i = 0; i < 3; i++) res.Get(i, 3) = aScale.Get(i);
		res.Get(3, 3) = 1;
		return res;
	}

	static Mat4f Perspective(
		float aFov,
		float aNear,
		float aFar)
	{
		// Camera points towards -z.  0 < near < far.
		// Matrix maps z range [-near, -far] to [-1, 1], after homogeneous division.
		float f = 1.f / (std::tan(aFov * PI_F / 360.0f));
		float d = 1.f / (aNear - aFar);

		Mat4f r;
		r.m00 = f;    r.m01 = 0.0f; r.m02 = 0.0f;               r.m03 = 0.0f;
		r.m10 = 0.0f; r.m11 = -f;   r.m12 = 0.0f;               r.m13 = 0.0f;
		r.m20 = 0.0f; r.m21 = 0.0f; r.m22 = (aNear + aFar) * d; r.m23 = 2.0f * aNear * aFar * d;
		r.m30 = 0.0f; r.m31 = 0.0f; r.m32 = -1.0f;              r.m33 = 0.0f;

		return r;
	}
public:

	// m_row_col; stored column major
	float m00, m10, m20, m30;
	float m01, m11, m21, m31;
	float m02, m12, m22, m32;
	float m03, m13, m23, m33;
};

Mat4f operator*(const Mat4f& left, const Mat4f& right)
{
	Mat4f res(0);
	for (int row = 0; row < 4; row++)
		for (int col = 0; col < 4; col++)
			for (int i = 0; i < 4; i++)
				res.Get(row, col) += left.Get(row, i) * right.Get(i, col);

	return res;
}

// Code for inversion taken from:
// http://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
Mat4f Invert(const Mat4f& aMatrix)
{
	const float *m = aMatrix.GetPtr();
	float inv[16], det;
	int i;

	inv[0] = m[5] * m[10] * m[15] -
		m[5] * m[11] * m[14] -
		m[9] * m[6] * m[15] +
		m[9] * m[7] * m[14] +
		m[13] * m[6] * m[11] -
		m[13] * m[7] * m[10];

	inv[4] = -m[4] * m[10] * m[15] +
		m[4] * m[11] * m[14] +
		m[8] * m[6] * m[15] -
		m[8] * m[7] * m[14] -
		m[12] * m[6] * m[11] +
		m[12] * m[7] * m[10];

	inv[8] = m[4] * m[9] * m[15] -
		m[4] * m[11] * m[13] -
		m[8] * m[5] * m[15] +
		m[8] * m[7] * m[13] +
		m[12] * m[5] * m[11] -
		m[12] * m[7] * m[9];

	inv[12] = -m[4] * m[9] * m[14] +
		m[4] * m[10] * m[13] +
		m[8] * m[5] * m[14] -
		m[8] * m[6] * m[13] -
		m[12] * m[5] * m[10] +
		m[12] * m[6] * m[9];

	inv[1] = -m[1] * m[10] * m[15] +
		m[1] * m[11] * m[14] +
		m[9] * m[2] * m[15] -
		m[9] * m[3] * m[14] -
		m[13] * m[2] * m[11] +
		m[13] * m[3] * m[10];

	inv[5] = m[0] * m[10] * m[15] -
		m[0] * m[11] * m[14] -
		m[8] * m[2] * m[15] +
		m[8] * m[3] * m[14] +
		m[12] * m[2] * m[11] -
		m[12] * m[3] * m[10];

	inv[9] = -m[0] * m[9] * m[15] +
		m[0] * m[11] * m[13] +
		m[8] * m[1] * m[15] -
		m[8] * m[3] * m[13] -
		m[12] * m[1] * m[11] +
		m[12] * m[3] * m[9];

	inv[13] = m[0] * m[9] * m[14] -
		m[0] * m[10] * m[13] -
		m[8] * m[1] * m[14] +
		m[8] * m[2] * m[13] +
		m[12] * m[1] * m[10] -
		m[12] * m[2] * m[9];

	inv[2] = m[1] * m[6] * m[15] -
		m[1] * m[7] * m[14] -
		m[5] * m[2] * m[15] +
		m[5] * m[3] * m[14] +
		m[13] * m[2] * m[7] -
		m[13] * m[3] * m[6];

	inv[6] = -m[0] * m[6] * m[15] +
		m[0] * m[7] * m[14] +
		m[4] * m[2] * m[15] -
		m[4] * m[3] * m[14] -
		m[12] * m[2] * m[7] +
		m[12] * m[3] * m[6];

	inv[10] = m[0] * m[5] * m[15] -
		m[0] * m[7] * m[13] -
		m[4] * m[1] * m[15] +
		m[4] * m[3] * m[13] +
		m[12] * m[1] * m[7] -
		m[12] * m[3] * m[5];

	inv[14] = -m[0] * m[5] * m[14] +
		m[0] * m[6] * m[13] +
		m[4] * m[1] * m[14] -
		m[4] * m[2] * m[13] -
		m[12] * m[1] * m[6] +
		m[12] * m[2] * m[5];

	inv[3] = -m[1] * m[6] * m[11] +
		m[1] * m[7] * m[10] +
		m[5] * m[2] * m[11] -
		m[5] * m[3] * m[10] -
		m[9] * m[2] * m[7] +
		m[9] * m[3] * m[6];

	inv[7] = m[0] * m[6] * m[11] -
		m[0] * m[7] * m[10] -
		m[4] * m[2] * m[11] +
		m[4] * m[3] * m[10] +
		m[8] * m[2] * m[7] -
		m[8] * m[3] * m[6];

	inv[11] = -m[0] * m[5] * m[11] +
		m[0] * m[7] * m[9] +
		m[4] * m[1] * m[11] -
		m[4] * m[3] * m[9] -
		m[8] * m[1] * m[7] +
		m[8] * m[3] * m[5];

	inv[15] = m[0] * m[5] * m[10] -
		m[0] * m[6] * m[9] -
		m[4] * m[1] * m[10] +
		m[4] * m[2] * m[9] +
		m[8] * m[1] * m[6] -
		m[8] * m[2] * m[5];

	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

	if (det == 0)
		return Mat4f::Identity();

	det = 1.f / det;

	Mat4f res;
	for (i = 0; i < 16; i++)
		res.GetPtr()[i] = inv[i] * det;

	return res;
}

float Luminance(const Vec3f& aRGB)
{
	return 0.212671f * aRGB.x +
		0.715160f * aRGB.y +
		0.072169f * aRGB.z;
}

float FresnelDielectric(
	float aCosInc,
	float IOR)
{
	if (IOR < 0)
		return 1.f;

	float eta;

	if (aCosInc < 0.f)
	{
		aCosInc = -aCosInc;
		eta = IOR;
	}
	else
	{
		eta = 1.f / IOR;
	}

	const float sinTrans2 = Sqr(eta) * (1.f - Sqr(aCosInc));
	const float cosTrans = std::sqrt(std::max(0.f, 1.f - sinTrans2));

	const float term1 = eta * cosTrans;
	const float rParallel =
		(aCosInc - term1) / (aCosInc + term1);

	const float term2 = eta * aCosInc;
	const float rPerpendicular =
		(term2 - cosTrans) / (term2 + cosTrans);

	return 0.5f * (Sqr(rParallel) + Sqr(rPerpendicular));
}

Vec3f ReflectLocal(const Vec3f& aVector)
{
	return Vec3f(-aVector.x, -aVector.y, aVector.z);
}

//////////////////////////////////////////////////////////////////////////
// Cosine lobe hemisphere sampling
Vec3f SamplePowerCosHemisphereW(
	const Vec2f  &aSamples,
	const float  aPower,
	float        *oPdfW)
{
	const float term1 = 2.f * PI_F * aSamples.x;
	const float term2 = std::pow(aSamples.y, 1.f / (aPower + 1.f));
	const float term3 = std::sqrt(1.f - term2 * term2);

	if (oPdfW)
	{
		*oPdfW = (aPower + 1.f) * std::pow(term2, aPower) * (0.5f * INV_PI_F);
	}

	return Vec3f(
		std::cos(term1) * term3,
		std::sin(term1) * term3,
		term2);
}

float PowerCosHemispherePdfW(
	const Vec3f  &aNormal,
	const Vec3f  &aDirection,
	const float  aPower)
{
	const float cosTheta = std::max(0.f, Dot(aNormal, aDirection));

	return (aPower + 1.f) * std::pow(cosTheta, aPower) * (INV_PI_F * 0.5f);
}

//////////////////////////////////////////////////////////////////////////
/// Sample direction in the upper hemisphere with cosine-proportional pdf
/** The returned PDF is with respect to solid angle measure */
Vec3f SampleCosHemisphereW(
	const Vec2f  &aSamples,
	float        *oPdfW)
{
	const float term1 = 2.f * PI_F * aSamples.x;
	const float term2 = std::sqrt(1.f - aSamples.y);

	const Vec3f ret(
		std::cos(term1) * term2,
		std::sin(term1) * term2,
		std::sqrt(aSamples.y));

	if (oPdfW)
	{
		*oPdfW = ret.z * INV_PI_F;
	}

	return ret;
}

// Sample Triangle
// returns barycentric coordinates
Vec2f SampleUniformTriangle(const Vec2f &aSamples)
{
	const float term = std::sqrt(aSamples.x);

	return Vec2f(1.f - term, aSamples.y * term);
}

float PdfAtoW(
	const float aPdfA,
	const float aDist,
	const float aCosThere)
{
	return aPdfA * Sqr(aDist) / std::abs(aCosThere);
}
#pragma endregion MATHUTIL

//////////////////////////////////////////////////////////////////////////
// Random Number Generator = RNG
class RNG
{
public:
	RNG(int aSeed = 1234) : randGen(aSeed) {}

	float GetFloat()
	{
		return mDistFloat(randGen);
	}

	Vec2f GetVec2f()
	{
		return Vec2f(GetFloat(), GetFloat());
	}

	Vec3f GetVec3f()
	{
		return Vec3f(GetFloat(), GetFloat(), GetFloat());
	}

private:
	std::mt19937_64 randGen;
	std::uniform_real_distribution<float> mDistFloat;
};

//////////////////////////////////////////////////////////////////////////
// BMP TOOL
// References: http://www.smallvcm.com
#pragma region BMPUTIL
class Frame
{
public:

	Frame()
	{
		mX = Vec3f(1, 0, 0);
		mY = Vec3f(0, 1, 0);
		mZ = Vec3f(0, 0, 1);
	};

	Frame(const Vec3f& x, const Vec3f& y, const Vec3f& z) : mX(x), mY(y), mZ(z) {}

	void SetFromZ(const Vec3f& z)
	{
		Vec3f tmpZ = mZ = Normalize(z);
		Vec3f tmpX = (std::abs(tmpZ.x) > 0.99f) ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0);
		mY = Normalize(Cross(tmpZ, tmpX));
		mX = Cross(mY, tmpZ);
	}

	Vec3f ToWorld(const Vec3f& a) const
	{
		return mX * a.x + mY * a.y + mZ * a.z;
	}

	Vec3f ToLocal(const Vec3f& a) const
	{
		return Vec3f(Dot(a, mX), Dot(a, mY), Dot(a, mZ));
	}

public:
	Vec3f mX, mY, mZ;
};

class Framebuffer
{
public:

	Framebuffer()
	{}

	void AddColor(
		const Vec2f& aSample,
		const Vec3f& aColor)
	{
		if (aSample.x < 0 || aSample.x >= resX)
			return;

		if (aSample.y < 0 || aSample.y >= resY)
			return;

		int x = int(aSample.x);
		int y = int(aSample.y);

		mColor[x + y * resX] = mColor[x + y * resX] + aColor;
	}

	void Setup(const Vec2f& aResolution)
	{
		resX = int(aResolution.x);
		resY = int(aResolution.y);
		mColor.resize(resX * resY);
		memset(&mColor[0], 0, sizeof(Vec3f) * mColor.size());
	}

	void Scale(float aScale)
	{
		for (size_t i = 0; i < mColor.size(); i++)
			mColor[i] = mColor[i] * Vec3f(aScale);
	}

	// Saving BMP
	struct BmpHeader
	{
		uint   mFileSize;        // Size of file in bytes
		uint   mReserved01;      // 2x 2 reserved bytes
		uint   mDataOffset;      // Offset in bytes where data can be found (54)

		uint   mHeaderSize;      // 40B
		int    mWidth;           // Width in pixels
		int    mHeight;          // Height in pixels

		short  mColorPlates;     // Must be 1
		short  mBitsPerPixel;    // We use 24bpp
		uint   mCompression;     // We use BI_RGB ~ 0, uncompressed
		uint   mImageSize;       // mWidth x mHeight x 3B
		uint   mHorizRes;        // Pixels per meter (75dpi ~ 2953ppm)
		uint   mVertRes;         // Pixels per meter (75dpi ~ 2953ppm)
		uint   mPaletteColors;   // Not using palette - 0
		uint   mImportantColors; // 0 - all are important
	};

	void SaveBMP(const char *aFilename)
	{
		std::ofstream bmp(aFilename, std::ios::binary);
		BmpHeader header;
		bmp.write("BM", 2);
		header.mFileSize = uint(sizeof(BmpHeader) + 2) + resX * resY * 3;
		header.mReserved01 = 0;
		header.mDataOffset = uint(sizeof(BmpHeader) + 2);
		header.mHeaderSize = 40;
		header.mWidth = resX;
		header.mHeight = resY;
		header.mColorPlates = 1;
		header.mBitsPerPixel = 24;
		header.mCompression = 0;
		header.mImageSize = resX * resY * 3;
		header.mHorizRes = 2953;
		header.mVertRes = 2953;
		header.mPaletteColors = 0;
		header.mImportantColors = 0;

		bmp.write((char*)&header, sizeof(header));

		const float gamma = 0.55f;
		for (int y = 0; y < resY; y++)
		{
			for (int x = 0; x < resX; x++)
			{
				// bmp is stored from bottom up
				const Vec3f &rgbF = mColor[x + (resY - y - 1)*resX];
				typedef unsigned char byte;
				float gammaBgr[3];
				gammaBgr[0] = std::pow(rgbF.z, gamma) * 255.f;
				gammaBgr[1] = std::pow(rgbF.y, gamma) * 255.f;
				gammaBgr[2] = std::pow(rgbF.x, gamma) * 255.f;

				byte bgrB[3];
				bgrB[0] = byte(std::min(255.f, std::max(0.f, gammaBgr[0])));
				bgrB[1] = byte(std::min(255.f, std::max(0.f, gammaBgr[1])));
				bgrB[2] = byte(std::min(255.f, std::max(0.f, gammaBgr[2])));

				bmp.write((char*)&bgrB, sizeof(bgrB));
			}
		}
	}

private:

	std::vector<Vec3f> mColor;
	int                resX;
	int                resY;
};
#pragma endregion BMPUTIL

//////////////////////////////////////////////////////////////////////////
// Materials
class Material
{
public:
	Material()
	{
		Reset();
	}

	void Reset()
	{
		diffuseFact = Vec3f(0);
		PhongFact = Vec3f(0);
		PhongExp = 1.f;
		mirrorFact = Vec3f(0);
		IOR = -1.f;
	}

public:
	Vec3f diffuseFact;
	Vec3f PhongFact;
	float PhongExp;
	Vec3f mirrorFact;
	float IOR;
};

class Ray
{
public:
	Ray()
	{}

	Ray(const Vec3f& aOrg, const Vec3f& aDir) : org(aOrg), dir(aDir) {}

public:
	Vec3f org;  //!< Ray origin
	Vec3f dir;  //!< Ray direction
};

class Intersection
{
public:
	Intersection()
	{}

	Intersection(float aMaxDist) :dist(aMaxDist)
	{}

public:
	float dist;    //!< Distance to closest intersection (serves as ray.tmax)
	int   matID;   //!< ID of intersected material
	Vec3f normal;  //!< Normal at the intersection
};

//////////////////////////////////////////////////////////////////////////
// Geometry
// References: http://www.smallvcm.com
#pragma region GEOMETRY
class AbstractGeometry
{
public:

	virtual ~AbstractGeometry() {};

	// Finds the closest intersection
	virtual bool Intersect(const Ray& aRay, Intersection& oResult) const = 0;

	// Finds any intersection, default calls Intersect
	virtual bool IntersectP(const Ray& aRay, Intersection& oResult) const
	{
		return Intersect(aRay, oResult);
	}

	// Grows given bounding box by this object
	virtual void GrowBBox(Vec3f &aoBBoxMin, Vec3f &aoBBoxMax) = 0;
};

class GeometryList : public AbstractGeometry
{
public:

	virtual ~GeometryList()
	{
		for (int i = 0; i < (int)geomtry.size(); i++)
			delete geomtry[i];
	};

	virtual bool Intersect(const Ray& aRay, Intersection& oResult) const
	{
		bool anyIntersection = false;

		for (int i = 0; i < (int)geomtry.size(); i++)
		{
			bool hit = geomtry[i]->Intersect(aRay, oResult);

			if (hit)
				anyIntersection = hit;
		}

		return anyIntersection;
	}

	virtual bool IntersectP(
		const Ray &aRay,
		Intersection     &oResult) const
	{
		for (int i = 0; i < (int)geomtry.size(); i++)
		{
			if (geomtry[i]->IntersectP(aRay, oResult))
				return true;
		}

		return false;
	}

	virtual void GrowBBox(
		Vec3f &aoBBoxMin,
		Vec3f &aoBBoxMax)
	{
		for (int i = 0; i < (int)geomtry.size(); i++)
			geomtry[i]->GrowBBox(aoBBoxMin, aoBBoxMax);
	}

public:

	std::vector<AbstractGeometry*> geomtry;
};

class Triangle : public AbstractGeometry
{
public:

	Triangle() {}

	Triangle(
		const Vec3f &p0,
		const Vec3f &p1,
		const Vec3f &p2,
		int         aMatID)
	{
		p[0] = p0;
		p[1] = p1;
		p[2] = p2;
		matID = aMatID;
		mNormal = Normalize(Cross(p[1] - p[0], p[2] - p[0]));
	}

	virtual bool Intersect(
		const Ray &aRay,
		Intersection     &oResult) const
	{
		const Vec3f ao = p[0] - aRay.org;
		const Vec3f bo = p[1] - aRay.org;
		const Vec3f co = p[2] - aRay.org;

		const Vec3f v0 = Cross(co, bo);
		const Vec3f v1 = Cross(bo, ao);
		const Vec3f v2 = Cross(ao, co);

		const float v0d = Dot(v0, aRay.dir);
		const float v1d = Dot(v1, aRay.dir);
		const float v2d = Dot(v2, aRay.dir);

		if (((v0d < 0.f) && (v1d < 0.f) && (v2d < 0.f)) ||
			((v0d >= 0.f) && (v1d >= 0.f) && (v2d >= 0.f)))
		{
			const float distance = Dot(mNormal, ao) / Dot(mNormal, aRay.dir);

			if ((distance > 0) && (distance < oResult.dist))
			{
				oResult.normal = mNormal;
				oResult.matID = matID;
				oResult.dist = distance;
				return true;
			}
		}

		return false;
	}

	virtual void GrowBBox(
		Vec3f &aoBBoxMin,
		Vec3f &aoBBoxMax)
	{
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				aoBBoxMin.Get(j) = std::min(aoBBoxMin.Get(j), p[i].Get(j));
				aoBBoxMax.Get(j) = std::max(aoBBoxMax.Get(j), p[i].Get(j));
			}
		}
	}

public:

	Vec3f p[3];
	int   matID;
	Vec3f mNormal;
};

class Sphere : public AbstractGeometry
{
public:

	Sphere() {}

	Sphere(
		const Vec3f &aCenter,
		float       aRadius,
		int         aMatID)
	{
		center = aCenter;
		radius = aRadius;
		matID = aMatID;
	}

	// Taken from:
	// http://wiki.cgsociety.org/index.php/Ray_Sphere_Intersection

	virtual bool Intersect(
		const Ray &aRay,
		Intersection     &oResult) const
	{
		// we transform ray origin into object space (center == origin)
		const Vec3f transformedOrigin = aRay.org - center;

		const float A = Dot(aRay.dir, aRay.dir);
		const float B = 2 * Dot(aRay.dir, transformedOrigin);
		const float C = Dot(transformedOrigin, transformedOrigin) - (radius * radius);

		// Must use doubles, because when B ~ sqrt(B*B - 4*A*C)
		// the resulting t is imprecise enough to get around ray epsilons
		const double disc = B*B - 4 * A*C;

		if (disc < 0)
			return false;

		const double discSqrt = std::sqrt(disc);
		const double q = (B < 0) ? ((-B - discSqrt) / 2.f) : ((-B + discSqrt) / 2.f);

		double t0 = q / A;
		double t1 = C / q;

		if (t0 > t1) std::swap(t0, t1);

		float resT;

		if (t0 > 0 && t0 < oResult.dist)
			resT = float(t0);
		else if (t1 > 0 && t1 < oResult.dist)
			resT = float(t1);
		else
			return false;

		oResult.dist = resT;
		oResult.matID = matID;
		oResult.normal = Normalize(transformedOrigin + Vec3f(resT) * aRay.dir);
		return true;
	}

	virtual void GrowBBox(
		Vec3f &aoBBoxMin,
		Vec3f &aoBBoxMax)
	{
		for (int i = 0; i < 8; i++)
		{
			Vec3f p = center;
			Vec3f half(radius);

			for (int j = 0; j < 3; j++)
				if (i & (1 << j)) half.Get(j) = -half.Get(j);

			p += half;

			for (int j = 0; j < 3; j++)
			{
				aoBBoxMin.Get(j) = std::min(aoBBoxMin.Get(j), p.Get(j));
				aoBBoxMax.Get(j) = std::max(aoBBoxMax.Get(j), p.Get(j));
			}
		}
	}

public:

	Vec3f center;
	float radius;
	int   matID;
};
#pragma endregion GEOMETRY

//////////////////////////////////////////////////////////////////////////
// Sun light
class Light
{
public:
	Light(const Vec3f& direct, const Vec3f& intens)
	{
		frame.SetFromZ(direct);
		intensity = intens;
	}

public:
	Frame frame;
	Vec3f intensity;
};

//////////////////////////////////////////////////////////////////////////
// My Camera
#pragma region CAMERA
class Camera
{
public:

	void Setup(const Vec3f &position, const Vec3f &forward, const Vec3f &up, const Vec2f &resolution)
	{
		const Vec3f forwardN = Normalize(forward);
		const Vec3f upN = Normalize(Cross(up, -forwardN));
		const Vec3f leftN = Cross(-forwardN, upN);

		pos = position;
		face = forwardN;
		res = resolution;

		const Vec3f pos(
			Dot(upN, position),
			Dot(leftN, position),
			Dot(-forwardN, position));

		Mat4f worldToCamera = Mat4f::Identity();
		worldToCamera.SetRow(0, upN, -pos.x);
		worldToCamera.SetRow(1, leftN, -pos.y);
		worldToCamera.SetRow(2, -forwardN, -pos.z);

		const Mat4f perspective = Mat4f::Perspective(45, 0.1f, 10000.f);
		const Mat4f worldToScreen = perspective * worldToCamera;
		const Mat4f screenToWorld = Invert(worldToScreen);

		toWorldMatrix = screenToWorld * Mat4f::Translate(Vec3f(-1.f, -1.f, 0)) * Mat4f::Scale(Vec3f(2.f / resolution.x, 2.f / resolution.y, 0));
	}

	Vec3f RasterToWorld(const Vec2f &aRasterXY) const
	{
		return toWorldMatrix.TransformPoint(Vec3f(aRasterXY.x, aRasterXY.y, 0));
	}

	Ray GenerateRay(const Vec2f &aRasterXY) const
	{
		const Vec3f worldRaster = RasterToWorld(aRasterXY);

		Ray res;
		res.org = pos;
		res.dir = Normalize(worldRaster - pos);
		return res;
	}
public:

	Vec3f pos;
	Vec3f face;
	Vec2f res;
	Mat4f toWorldMatrix;
};
#pragma endregion CAMERA

//////////////////////////////////////////////////////////////////////////
// My Scene
#pragma region MYSCENE
class Scene
{
public:
	Scene() :
		geo(nullptr)
	{}

	~Scene()
	{
		delete geo;
		delete sun;
	}

	bool Intersect(const Ray & ray, Intersection & intersectRes) const
	{
		return geo->Intersect( ray, intersectRes );
	}

	bool Occluded(
		const Vec3f &target,
		const Vec3f &direction,
		float maxDist) const
	{
		Ray ray;
		ray.org = target + direction * EPS_RAY;
		ray.dir = direction;
		Intersection isect;
		isect.dist = maxDist - 2 * EPS_RAY;

		return geo->IntersectP(ray, isect);
	}

	void BuildBox(const Vec2i &resolution) // waiting for done
	{
		Vec3f camPos = Vec3f(-0.0439815f, -5.22529f, 0.222539f);
		Vec3f camFace = Vec3f(0.00688625f, 0.998505f, -0.0542161f);
		Vec3f camUp = Vec3f(3.73896e-4f, 0.0542148f, 0.998529f);

		cam.Setup(camPos, camFace, camUp, Vec2f(float(resolution.x), float(resolution.y)));

		delete geo;
		delete sun;

		// Materials
		Material mat;
		// 0) glossy white floor
		mat.Reset();
		mat.diffuseFact = Vec3f(0.1f);
		mat.PhongFact = Vec3f(0.7f);
		mat.PhongExp = 90.f;
		matVec.push_back(mat);

		// 1) diffuse yellow wall
		mat.Reset();
		mat.diffuseFact = Vec3f(0.996863f, 0.903922f, 0.172549f);
		matVec.push_back(mat);

		// 2) diffuse red wall
		mat.Reset();
		mat.diffuseFact = Vec3f(0.803922f, 0.152941f, 0.152941f);
		matVec.push_back(mat);

		// 3) diffuse blue wall
		mat.Reset();
		mat.diffuseFact = Vec3f(0.203922f, 0.703922f, 0.703922f);
		matVec.push_back(mat);

		// 4) mirror ball
		mat.Reset();
		mat.mirrorFact = Vec3f(1.f);
		matVec.push_back(mat);

		// 5) glass ball
		mat.Reset();
		mat.mirrorFact = Vec3f(1.f);
		mat.IOR = 1.6f;
		matVec.push_back(mat);

		// 6) diffuse dark blue wall
		mat.Reset();
		mat.diffuseFact = Vec3f(0.056863f, 0.412549f, 0.633922f);
		matVec.push_back(mat);

		// 7) diffuse light 
		mat.Reset();
		mat.diffuseFact = Vec3f(0.8f, 0.8f, 0.8f);
		matVec.push_back(mat);


		//////////////////////////////////////////////////////////////////////////
		// Cornell box
		Vec3f cb[8] = {
			Vec3f(-1.77029f,  1.30455f, -1.68000f),
			Vec3f(1.88975f,  1.30455f, -1.68000f),
			Vec3f(1.88975f,  1.30455f,  1.68000f),
			Vec3f(-1.77029f,  1.30455f,  1.68000f),
			Vec3f(-1.77029f, -1.25549f, -1.68000f),
			Vec3f(1.88975f, -1.25549f, -1.68000f),
			Vec3f(1.88975f, -1.25549f,  1.68000f),
			Vec3f(-1.77029f, -1.25549f,  1.68000f)
		};

		GeometryList *geometryList = new GeometryList;
		geo = geometryList;

		// Floor
		geometryList->geomtry.push_back(new Triangle(cb[0], cb[4], cb[5], 0));
		geometryList->geomtry.push_back(new Triangle(cb[5], cb[1], cb[0], 0));

		// Back wall
		geometryList->geomtry.push_back(new Triangle(cb[0], cb[1], cb[2], 3));
		geometryList->geomtry.push_back(new Triangle(cb[2], cb[3], cb[0], 3));

		// Ceiling
		geometryList->geomtry.push_back(new Triangle(cb[2], cb[6], cb[7], 6));
		geometryList->geomtry.push_back(new Triangle(cb[7], cb[3], cb[2], 6));

		// Left wall
		geometryList->geomtry.push_back(new Triangle(cb[3], cb[7], cb[4], 2));
		geometryList->geomtry.push_back(new Triangle(cb[4], cb[0], cb[3], 2));

		// Right wall
		geometryList->geomtry.push_back(new Triangle(cb[1], cb[5], cb[6], 1));
		geometryList->geomtry.push_back(new Triangle(cb[6], cb[2], cb[1], 1));

		// cube 1
		Vec3f cube[8] = {
			Vec3f(-1.0f,  0.4f, -3.f),
			Vec3f(-0.2f,  0.8f, -3.f),
			Vec3f(-0.2f,  0.8f,  0.0f),
			Vec3f(-1.0f,  0.4f,  0.0f),
			Vec3f(-0.6f, -0.0f, -3.f),
			Vec3f(0.2f, 0.4f, -3.f),
			Vec3f(0.2f, 0.4f,  0.0f),
			Vec3f(-0.6f, -0.0f,  0.0f)
		};

		int matID = 7;
		geometryList->geomtry.push_back(new Triangle(cube[0], cube[3], cube[2], matID));
		geometryList->geomtry.push_back(new Triangle(cube[2], cube[1], cube[0], matID));
		geometryList->geomtry.push_back(new Triangle(cube[2], cube[3], cube[7], matID));
		geometryList->geomtry.push_back(new Triangle(cube[7], cube[6], cube[2], matID));
		geometryList->geomtry.push_back(new Triangle(cube[3], cube[0], cube[4], matID));
		geometryList->geomtry.push_back(new Triangle(cube[4], cube[7], cube[3], matID));
		geometryList->geomtry.push_back(new Triangle(cube[1], cube[2], cube[6], matID));
		geometryList->geomtry.push_back(new Triangle(cube[6], cube[5], cube[1], matID));
		geometryList->geomtry.push_back(new Triangle(cube[4], cube[5], cube[6], matID));
		geometryList->geomtry.push_back(new Triangle(cube[6], cube[7], cube[4], matID));

		// cube 2
		Vec3f cube_2[8] = {
			Vec3f(1.0f,  -0.4f, -3.f),
			Vec3f(1.4f,  -0.8f, -3.f),

			Vec3f(1.4f,  -0.8f,  -0.7f),
			Vec3f(1.0f,  -0.4f,  -0.7f),

			Vec3f(0.4f, -1.0f, -3.f),
			Vec3f(0.8f, -1.4f, -3.f),

			Vec3f(0.8f, -1.4f,  -0.7f),
			Vec3f(0.4f, -1.0f,  -0.7f)
		};

		matID = 5;

		geometryList->geomtry.push_back(new Triangle(cube_2[0], cube_2[3], cube_2[2], matID));
		geometryList->geomtry.push_back(new Triangle(cube_2[2], cube_2[1], cube_2[0], matID));
		geometryList->geomtry.push_back(new Triangle(cube_2[2], cube_2[3], cube_2[7], matID));
		geometryList->geomtry.push_back(new Triangle(cube_2[7], cube_2[6], cube_2[2], matID));
		geometryList->geomtry.push_back(new Triangle(cube_2[3], cube_2[0], cube_2[4], matID));
		geometryList->geomtry.push_back(new Triangle(cube_2[4], cube_2[7], cube_2[3], matID));
		geometryList->geomtry.push_back(new Triangle(cube_2[1], cube_2[2], cube_2[6], matID));
		geometryList->geomtry.push_back(new Triangle(cube_2[6], cube_2[5], cube_2[1], matID));
		geometryList->geomtry.push_back(new Triangle(cube_2[4], cube_2[5], cube_2[6], matID));
		geometryList->geomtry.push_back(new Triangle(cube_2[6], cube_2[7], cube_2[4], matID));

		// Balls
		float smallRadius = 0.2f;
		Vec3f leftWallCenter = (cb[0] + cb[4]) * (1.f / 2.f) + Vec3f(0, 0, smallRadius);
		Vec3f rightWallCenter = (cb[1] + cb[5]) * (1.f / 2.f) + Vec3f(0, 0, smallRadius);

		float xlen = rightWallCenter.x - leftWallCenter.x;
		Vec3f leftBallCenter = leftWallCenter + Vec3f(1.f * xlen / 7.f, 0.7f, 0);
		Vec3f rightBallCenter = rightWallCenter - Vec3f(1.f * xlen / 7.f + 0.1f, 0, -3.f * xlen / 7.f);
		Vec3f ThirdBallCenter = leftWallCenter + Vec3f(1.f * xlen / 7.f, -0.9f, 4.f * xlen / 7.f);

		geometryList->geomtry.push_back(new Sphere(rightBallCenter, 0.5f, 4));
		geometryList->geomtry.push_back(new Sphere(ThirdBallCenter, 0.3f, 5));
		geometryList->geomtry.push_back(new Sphere(leftBallCenter, 0.2, 4));
		leftBallCenter -= Vec3f(0.0f, 0.5f, 0);

		geometryList->geomtry.push_back(new Sphere(leftBallCenter, 0.2, 4));
		Vec3f line_1 = leftBallCenter - Vec3f(0.2f, 0.4f, 0);
		geometryList->geomtry.push_back(new Sphere(line_1, 0.2, 5));
		Vec3f line_2 = line_1 - Vec3f(-0.2f, 0.7f, 0);
		geometryList->geomtry.push_back(new Sphere(line_2, 0.2, 5));
		line_1 += Vec3f(0.6f, -0.1f, 0);
		geometryList->geomtry.push_back(new Sphere(line_1, 0.2, 4));
		line_2 += Vec3f(0.8f, 0.0f, 0);
		geometryList->geomtry.push_back(new Sphere(line_2, 0.2, 5));
		line_1 += Vec3f(0.9f, -0.0f, 0);
		geometryList->geomtry.push_back(new Sphere(line_1, 0.2, 4));
		line_1 += Vec3f(1.1f, 0.3f, 0);
		geometryList->geomtry.push_back(new Sphere(line_1, 0.2, 4));
		line_1 += Vec3f(-0.4f, 0.5f, 0);
		geometryList->geomtry.push_back(new Sphere(line_1, 0.2, 5));

		// sun
		sun = new Light(Vec3f(-1.f, 1.5f, -1.f), Vec3f(0.4f, 0.4f, 0.4f) * 10.f);
	}

public:
	AbstractGeometry		*geo;
	Camera					cam;
	std::vector<Material>	matVec;
	Light					*sun;
};
#pragma endregion MYSCENE

//////////////////////////////////////////////////////////////////////////
// My BSDF Caculation
#pragma region BSDF
enum Events
{
	diffuse,
	Phong,
	reflect,
	refract,
};

class BSDF
{
	struct ComponentProbabilities
	{
		float diffProb;
		float phongProb;
		float reflectProb;
		float refractProb;
	};
public:
	BSDF(const Ray &ray, const Intersection &intersectRes, const Scene &scn)
	{
		matID = -1;
		frame.SetFromZ(intersectRes.normal);
		directionLFixed = frame.ToLocal(-ray.dir);

		if (std::abs(directionLFixed.z) < EPS_COSINE)
		{
			return;
		}

		matID = intersectRes.matID;
		const Material &mat = scn.matVec[matID];
		GetProbabilities(mat, probs);
	}

	Vec3f Evaluate(
		const Scene &scn,
		const Vec3f &directionW,
		float       &theta,
		float       *pdf) const
	{
		Vec3f result(0);
		*pdf = 0;

		const Vec3f directionL = frame.ToLocal(directionW);

		if (directionL.z * directionLFixed.z < 0)
			return result;

		theta = std::abs(directionL.z);

		const Material &mat = scn.matVec[matID];

		result += EvaluateDiffuse(mat, directionL, pdf);
		result += EvaluatePhong(mat, directionL, pdf);
		return result;
	}

	Vec3f Sample(
		const Scene &scn,
		const Vec3f &rn3,
		Vec3f       &directionW,
		float       &pdf,
		float       &theta) const
	{
		Events sampledEvent;
		pdf = 0;
		Vec3f result(0);
		Vec3f directionL;

		if (rn3.z < probs.diffProb)
			sampledEvent = diffuse;
		else if (rn3.z < probs.diffProb + probs.phongProb)
			sampledEvent = Phong;
		else if (rn3.z < probs.diffProb + probs.phongProb + probs.reflectProb)
			sampledEvent = reflect;
		else
			sampledEvent = refract;


		const Material &mat = scn.matVec[matID];

		switch (sampledEvent)
		{
		case diffuse:
		{
			result += SampleDiffuse(mat, rn3.GetXY(), directionL, pdf);

			if (result.IsZero())
				return Vec3f(0);

			result += EvaluatePhong(mat, directionL, &pdf);
			break;
		}
		case Phong:
		{
			result += SamplePhong(mat, rn3.GetXY(), directionL, pdf);

			if (result.IsZero())
				return Vec3f(0);

			result += EvaluateDiffuse(mat, directionL, &pdf);
			break;
		}
		case reflect:
		{
			result += SampleReflect(mat, directionL, pdf);

			if (result.IsZero())
				return Vec3f(0);
			break;
		}
		case refract:
		{
			result += SampleRefract(mat, directionL, pdf);

			if (result.IsZero())
				return Vec3f(0);
		}
		default:
			break;
		}

		theta = std::abs(directionL.z);

		if (theta < EPS_COSINE)
			return Vec3f(0.f);
		directionW = frame.ToWorld(directionL);
		return result;
	}

	float ContinuationProb() const
	{
		return probContinue;
	}

private:
	void GetProbabilities(const Material &mat, ComponentProbabilities &probs)
	{
		reflectCoeff = FresnelDielectric(directionLFixed.z, mat.IOR);

		const float albedoDiffuse = Luminance(mat.diffuseFact);
		const float albedoPhong = Luminance(mat.PhongFact);
		const float albedoReflect = reflectCoeff * Luminance(mat.mirrorFact);
		const float albedoRefract = (1.f - reflectCoeff) * (mat.IOR > 0.f ? 1.f : 0.f);
		const float totalAlbedo = albedoDiffuse + albedoPhong + albedoReflect + albedoRefract;

		probs.diffProb = albedoDiffuse / totalAlbedo;
		probs.phongProb = albedoPhong / totalAlbedo;
		probs.reflectProb = albedoReflect / totalAlbedo;
		probs.refractProb = albedoRefract / totalAlbedo;

		probContinue = (mat.diffuseFact + mat.PhongFact + reflectCoeff * mat.mirrorFact).Max() + (1.f - reflectCoeff);
		probContinue = std::min(1.f, std::max(0.f, probContinue));
	}

	Vec3f SampleDiffuse(
		const Material &mat,
		const Vec2f    &rn3,
		Vec3f          &directionL,
		float          &pdf) const
	{
		if (directionLFixed.z < EPS_COSINE)
			return Vec3f(0);

		float weight;
		directionL = SampleCosHemisphereW(rn3, &weight);
		pdf += weight * probs.diffProb;

		return mat.diffuseFact * INV_PI_F;
	}

	Vec3f SamplePhong(
		const Material &mat,
		const Vec2f    &rn3,
		Vec3f          &directionL,
		float          &pdf) const
	{
		if (probs.phongProb == 0)
			return Vec3f(0.f);

		directionL = SamplePowerCosHemisphereW(rn3, mat.PhongExp, NULL);

		const Vec3f reflLocalDirFixed = ReflectLocal(directionLFixed);
		
		Frame frame;
		frame.SetFromZ(reflLocalDirFixed);
		directionL = frame.ToWorld(directionL);

		float dot_R_Wi = Dot(reflLocalDirFixed, directionL);

		if (dot_R_Wi <= EPS_PHONG)
			return Vec3f(0.f);

		pdf += PowerCosHemispherePdfW(reflLocalDirFixed, directionL, mat.PhongExp) * probs.phongProb;
		const Vec3f rho = mat.PhongFact * (mat.PhongExp + 2.f) * 0.5f * INV_PI_F;

		return rho * std::pow(dot_R_Wi, mat.PhongExp);
	}

	Vec3f SampleReflect(
		const Material &mat,
		Vec3f          &directionL,
		float          &pdf) const
	{
		directionL = ReflectLocal(directionLFixed);

		pdf += probs.reflectProb;

		return reflectCoeff * mat.mirrorFact / std::abs(directionL.z);
	}

	Vec3f SampleRefract(
		const Material &mat,
		Vec3f          &directionL,
		float          &pdf) const
	{
		if (mat.IOR < 0)
			return Vec3f(0);

		float cosI = directionLFixed.z;
		float cosT;
		float etaOetaI;

		if (cosI < 0.f) 
		{
			etaOetaI = mat.IOR;
			cosI = -cosI;
			cosT = 1.f;
		}
		else
		{
			etaOetaI = 1.f / mat.IOR;
			cosT = -1.f;
		}

		const float sinI2 = 1.f - cosI * cosI;
		const float sinT2 = Sqr(etaOetaI) * sinI2;

		if (sinT2 < 1.f) 
		{
			cosT *= std::sqrt(1.f - sinT2);

			directionL = Vec3f(-etaOetaI * directionLFixed.x, -etaOetaI * directionLFixed.y, cosT);
			pdf += probs.refractProb;

			const float refractCoeff = 1.f - reflectCoeff;
			return Vec3f(refractCoeff * Sqr(etaOetaI) / std::abs(cosT));
		}
		return Vec3f(0.f);
	}

	Vec3f EvaluateDiffuse(
		const Material &mat,
		const Vec3f    &directionL,
		float          *pdf) const
	{
		if (probs.diffProb == 0)
			return Vec3f(0);

		if (directionLFixed.z < EPS_COSINE || directionL.z < EPS_COSINE)
			return Vec3f(0);

		*pdf += probs.diffProb * std::max(0.f, directionL.z * INV_PI_F);

		return mat.diffuseFact * INV_PI_F;
	}

	Vec3f EvaluatePhong(
		const Material &mat,
		const Vec3f    &directionL,
		float          *pdf) const
	{
		if (probs.phongProb == 0)
			return Vec3f(0);

		if (directionLFixed.z < EPS_COSINE || directionL.z < EPS_COSINE)
			return Vec3f(0);

		const Vec3f reflLocalDirIn = ReflectLocal(directionLFixed);
		const float dot_R_Wi = Dot(reflLocalDirIn, directionL);

		if (dot_R_Wi <= EPS_PHONG)
			return Vec3f(0.f);

		*pdf += probs.phongProb * PowerCosHemispherePdfW(reflLocalDirIn, directionL, mat.PhongExp);

		const Vec3f rho = mat.PhongFact * (mat.PhongExp + 2.f) * 0.5f * INV_PI_F;

		return rho * std::pow(dot_R_Wi, mat.PhongExp);
	}

private:    
	Vec3f directionLFixed;     
	ComponentProbabilities probs; 
	float probContinue;
	float reflectCoeff;     
	int   matID;
	Frame frame;
};
#pragma endregion BSDF

//////////////////////////////////////////////////////////////////////////
// My Renderer
#pragma region RENDERER
class LightTracer
{
public:

	LightTracer(const Scene& aScene) : scn(aScene)
	{
		maxPathLength = 10;
		iter = 0;
		randGen = int(rand() % 1000);
		buffer.Setup(aScene.cam.res);
	}

	~LightTracer() {}

	void RunIteration()
	{
		const int resX = int(scn.cam.res.x);
		const int resY = int(scn.cam.res.y);

		for (int pixID = 0; pixID < resX * resY; pixID++)
		{
			const int x = pixID % resX;
			const int y = pixID / resX;

			const Vec2f sample = Vec2f(float(x), float(y)) + randGen.GetVec2f();

			Ray ray = scn.cam.GenerateRay(sample);
			Intersection isect;

			isect.dist = 1e36f;
			Vec3f pathWeight(1.f);
			Vec3f color(0.f);
			uint pathLength = 1;

			for (;; ++pathLength)
			{
				scn.Intersect(ray, isect);
				Vec3f hitPoint = ray.org + ray.dir * isect.dist;
				BSDF bsdf(ray, isect, scn);

				if (pathLength >= maxPathLength || bsdf.ContinuationProb() == 0)
					break;

				const Light *light = scn.sun;

				Vec3f directionToLight = -light->frame.mZ;
				float distance = 1e36f;
				Vec3f radiance = light->intensity;		
				float bsdfPdfW, cosThetaOut;
				Vec3f factor = bsdf.Evaluate(scn, directionToLight, cosThetaOut, &bsdfPdfW);
					
				if (!factor.IsZero() && !scn.Occluded(hitPoint, directionToLight, distance))
				{
					Vec3f contrib = cosThetaOut * radiance * factor;
					color += pathWeight * contrib;
				}		

				Vec3f rn3 = randGen.GetVec3f();
				float pdf;

				Vec3f factorSample = bsdf.Sample(scn, rn3, ray.dir, pdf, cosThetaOut);
				if (factorSample.IsZero())
					break;

				float continueProb = bsdf.ContinuationProb();
				pdf *= std::min(1.f, continueProb);
				pathWeight *= factorSample * (cosThetaOut / pdf);
				ray.org = hitPoint + EPS_RAY * ray.dir;
				isect.dist = 1e36f;

			}
			buffer.AddColor(sample, color);
		}
		iter++;
	}

	void GetFramebuffer(Framebuffer& oFramebuffer)
	{
		oFramebuffer = buffer;

		if (iter > 0)
			oFramebuffer.Scale(1.f / iter);
	}

private:
	int maxPathLength;
	int iter;
	Framebuffer buffer;
	const Scene& scn;
	RNG randGen;
};
#pragma endregion RENDERER

float Render(const Scene& aScene, Framebuffer& aFramebuffer,const float& time)
{
	char x[300];
	memset(x, 0, 300);
	LightTracer renderer = LightTracer(aScene);

    clock_t startT = clock();
    
	while (clock() < startT + time*CLOCKS_PER_SEC)
	{
		renderer.RunIteration();
	}
    clock_t endT = clock();

	renderer.GetFramebuffer(aFramebuffer);

    return float(endT - startT) / CLOCKS_PER_SEC;
}

int main(int argc, const char *argv[])
{
	Scene *scene = new Scene;
	scene->BuildBox(Vec2i(512, 512));

	Framebuffer* buffer;
	buffer = new Framebuffer;

	float maxTime = 10;
 
    printf("Rendering my Cornell box: %g seconds render time\n", maxTime);
    printf("==============CG course Ray Tracing==============\n");

    float time = Render(*scene,*buffer,maxTime);
    printf("done in %.2f s\n", time);

    buffer->SaveBMP("myCornellBox.bmp");

    delete scene;
	delete buffer;

	printf("\n");
    return 0;
}
