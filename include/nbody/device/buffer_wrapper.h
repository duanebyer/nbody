#ifndef __NBODY_DEVICE_BUFFER_WRAPPER_H_
#define __NBODY_DEVICE_BUFFER_WRAPPER_H_

#include <algorithm>
#include <stdexcept>

#include "nbody/device/cl_includes.h"

#define BUFFER_RESIZE_FACTOR (2)

namespace nbody {
namespace device {

enum IOFlag {
	Read,
	Write,
	ReadWrite
};

class BufferWrapperException : std::exception {
	
private:
	
	std::string _what;
	
public:
	
	BufferWrapperException(std::string what) : _what(what) {
	}
	char const* what() const throw() {
		return _what.c_str();
	}
};

template<typename T>
class BufferWrapper {
	
private:
	
	cl::Context _context;
	cl::CommandQueue _queue;
	
	cl::Buffer _buffer;
	IOFlag _flag;
	std::size_t _size;
	std::size_t _capacity;
	
	cl_mem_flags memFlag(IOFlag flag) const {
		switch (flag) {
		case IOFlag::Read:
			return CL_MEM_READ_ONLY;
		case IOFlag::Write:
			return CL_MEM_WRITE_ONLY;
		case IOFlag::ReadWrite:
			return CL_MEM_READ_WRITE;
		default:
			return 0;
		}
	}
	
	cl_map_flags mapFlag(IOFlag flag) const {
		switch (flag) {
		case IOFlag::Read:
			return CL_MAP_READ;
		case IOFlag::Write:
			return CL_MAP_WRITE;
		case IOFlag::ReadWrite:
			return CL_MAP_READ | CL_MAP_WRITE;
		default:
			return 0;
		}
	}
	
	void reallocateBuffer(
			std::size_t size,
			std::size_t capacity,
			T const* data = NULL) {
		// Minimum capacity of OpenCL buffer is 1.
		if (capacity == 0) {
			capacity = 1;
		}
		// Determine whether the buffer should be initialized with data.
		bool shouldCopy = (data != NULL && capacity == size);
		bool shouldWrite = (data != NULL && !shouldCopy);
		cl_mem_flags memCopyFlag = (shouldCopy ?
			CL_MEM_COPY_HOST_PTR :
			CL_MEM_ALLOC_HOST_PTR);
		_buffer = cl::Buffer(
			_context,
			memFlag(_flag) | memCopyFlag,
			capacity * sizeof(T),
			const_cast<T*>(data));
		_size = size;
		_capacity = capacity;
		// Write data to buffer if we couldn't initialize the buffer with data.
		if (shouldWrite) {
			write(data);
		}
	}
	
public:
	
	BufferWrapper(
			cl::Context const& context,
			cl::CommandQueue const& queue,
			IOFlag flag,
			std::size_t size = 0,
			T const* data = NULL) :
			_context(context),
			_queue(queue),
			_flag(flag),
			_size(size),
			_capacity(size) {
		reallocateBuffer(_size, _capacity, data);
	}
	
	BufferWrapper(IOFlag flag) :
			_flag(flag) {
	}
	
	// Get basic information.
	std::size_t size() const {
		return _size;
	}
	std::size_t capacity() const {
		return _capacity;
	}
	IOFlag ioFlag() const {
		return _flag;
	}
	cl::Buffer buffer() {
		return _buffer;
	}
	
	// Resize buffers.
	void resize(
			std::size_t newSize,
			bool expandOnly = false,
			bool strict = false) {
		// Resize by factors of the BUFFER_RESIZE_FACTOR.
		std::size_t newCapacity;
		if (strict) {
			newCapacity = newSize;
		}
		else {
			newCapacity = _capacity;
			while (newCapacity < newSize) {
				newCapacity *= BUFFER_RESIZE_FACTOR;
			}
			while (newCapacity > BUFFER_RESIZE_FACTOR * newSize) {
				newCapacity /= BUFFER_RESIZE_FACTOR;
			}
		}
		
		// Make sure that the buffer only expands, if the flag is set.
		if (expandOnly) {
			newCapacity = std::max(_capacity, newCapacity);
		}
		
		// If there is no change in capacity, the size can simply be changed. 
		if (newCapacity == _capacity) {
			_size = newSize;
			return;
		}
		// Otherwise, have to reallocate the buffer.
		else {
			std::size_t oldSize = _size;
			cl::Buffer oldBuffer = std::move(_buffer);
			
			reallocateBuffer(newSize, newCapacity);
			
			std::size_t numCopy = std::min(_size, oldSize);
			if (numCopy != 0) {
				_queue.enqueueCopyBuffer(
					oldBuffer,
					_buffer,
					0, 0,
					numCopy * sizeof(T));
			}
		}
	}
	void reserve(std::size_t newCapacity) {
		if (newCapacity > _capacity) {
			cl::Buffer oldBuffer = std::move(_buffer);
			reallocateBuffer(_size, newCapacity);
			if (_size != 0) {
				_queue.enqueueCopyBuffer(
					oldBuffer,
					_buffer,
					0, 0,
					_size * sizeof(T));
			}
		}
	}
	
	// Map and unmap buffers.
	T* map(IOFlag flag) {
		if (_size == 0) {
			return NULL;
		}
		else {
			return reinterpret_cast<T*>(_queue.enqueueMapBuffer(
				_buffer,
				CL_TRUE,
				mapFlag(flag),
				0,
				_size * sizeof(T)));
		}
	}
	void unmap(T* data) {
		if (data != NULL) {
			_queue.enqueueUnmapMemObject(_buffer, data);
		}
	}
	
	// Read and write to buffers.
	void write(T const* data) {
		if (_size != 0) {
			_queue.enqueueWriteBuffer(
				_buffer,
				CL_TRUE,
				0,
				_size * sizeof(T),
				data);
		}
	}
	void read(T* data) {
		if (_size != 0) {
			_queue.enqueueReadBuffer(
				_buffer,
				CL_TRUE,
				0,
				_size * sizeof(T),
				data);
		}
	}
	void zero() {
		if (_size != 0) {
			_queue.enqueueFillBuffer(
				_buffer,
				static_cast<cl_uchar>(0),
				0,
				_size * sizeof(T));
		}
	}
	
	// Copy between buffers.
	void copyFrom(BufferWrapper<T> source) {
		std::size_t numCopy = std::min(_size, source._size);
		if (numCopy != 0) {
			_queue.enqueueCopyBuffer(
				source._buffer,
				_buffer,
				0, 0,
				numCopy * sizeof(T));
		}
	}
};

}
}

#endif

