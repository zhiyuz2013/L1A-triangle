# MIT License

# Copyright (c) 2026 Mingjie Zhang (mingjie2026@gmail.com)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

LOG = 0 # 10/29 2026

import gdb
import subprocess
import os
import sys
import numpy as np
import inspect
import math
import re

np.set_printoptions(suppress=True)  # FORMAT LONG G *
np.set_printoptions(linewidth=150) # MAX WIDTH NODELETE
np.set_printoptions(threshold=1000) # MAX NUMBER NODELETE
np.set_printoptions(edgeitems=10)   # FIRST LAST NUMBER NODELETE *
np.set_printoptions(formatter={
    'float_kind': lambda x: f'{x:8.3f}',
    'int_kind': lambda x: f'{x:3d}',
})
def printf(*args, **kwargs): print(*args, **kwargs)

PROJECT_ROOT = os.environ.get('PROJECT_ROOT')
if PROJECT_ROOT:              PATH = os.path.join(os.path.abspath(PROJECT_ROOT), 'mem')
elif '__file__' in globals(): PATH = str(Path(__file__).resolve().parent / 'mem')
else:                         PATH = os.path.abspath('mem')
os.makedirs(f'{PATH}/data',   exist_ok=True)
os.makedirs(f'{PATH}/figure', exist_ok=True)

def PRINT(*args, **kwargs):
    caller = inspect.currentframe().f_back
    try:
        print(f"{caller.f_lineno:3d}", f"{caller.f_code.co_name:<18}", *args, **kwargs)
    finally:
        del caller

TYPE_MAP = {
    'bool':                     np.uint8,
    'unsigned char':            np.uint8,
    'char':                     np.int8,
    'unsigned short':           np.uint16,      
    'short':                    np.int16,
    'unsigned int':             np.uint32,        
    'int':                      np.int32,
    'unsigned long':            np.dtype('L'),
    'unsigned long long':       np.uint64,
    'float':                    np.float32,
    'double':                   np.float64,       
}

def normalize_size(Size_mem, Size_read):   
    Size_mem = np.atleast_1d(np.asarray(Size_mem)) if Size_mem is not None else None
    Size_read = np.atleast_1d(np.asarray(Size_read)) if Size_read is not None else None

    if Size_read is None:
        return Size_mem

    Size_mem_p = math.prod(Size_mem)
    Size_read_p = math.prod(Size_read)

    if Size_mem_p >= Size_read_p:
        return Size_read

    Size = [1] * (len(Size_read))
    for i in range(len(Size_read) - 1, -1, -1):
        for j in range(Size_read[i]):
            Size_new = Size.copy()
            Size_new[i] = j + 1
            Size_new_p = math.prod(Size_new)

            if Size_new_p <= Size_mem_p:
                Size = Size_new
            else:
                break

    return Size

def cpu_memory(Addr, Type, Size):
    if LOG: PRINT('Addr', Addr, 'Type', Type, 'Size', Size)

    # A libstdc++ bit iterator identifies packed vector<bool> storage.
    if (Type == 'bool' and isinstance(Addr, gdb.Value)
            and Addr.type.strip_typedefs().code == gdb.TYPE_CODE_STRUCT):
        words = Addr['_M_p']
        bits = words.type.target().sizeof * 8
        offset = int(Addr['_M_offset'])
        length = math.prod(int(n) for n in np.atleast_1d(Size))
        values = [(int(words[(offset + i) // bits]) >> ((offset + i) % bits)) & 1
                  for i in range(length)]
        return np.array(values, dtype=np.uint8).reshape(Size)

    target = gdb.lookup_type(Type)   
    byte_count = target.sizeof * math.prod(Size)

    inferior = gdb.selected_inferior()
    raw = inferior.read_memory(Addr, byte_count)
    data = np.frombuffer(raw, dtype=TYPE_MAP[Type])

    data = data.reshape(Size)
    return data

def cpu_array(para, Type=None, Size=None):
    ftype = para.type.strip_typedefs()

    if Type is not None and Size is not None: # top priority
        Addr = int(para.address)
        return cpu_memory(Addr, Type, Size)  

    if Type is None:
        s = str(ftype.target().strip_typedefs().unqualified())
        Type = next((k for k, v in TYPE_MAP.items() if s.startswith(k)), None)

    # get array size line int[2][3]
    et, dims = ftype, []
    while et.code == gdb.TYPE_CODE_ARRAY:
        lo, hi = et.range()
        dims.append(hi - lo + 1)
        et = et.target().strip_typedefs().unqualified()

    if Size is None:
        Size = dims
    else:
        total_dims = int(np.prod(dims))
        total_size = int(np.prod(Size))
        if(total_dims < total_size):
            Size = dims        
        
    if LOG: PRINT('dims', dims)

    if Type is not None:
        Addr = int(para.address)

        if Type is None: # Type:None like int*
            Type = next((k for k, v in TYPE_MAP.items() if k in str(ftype)), None)
    
        if LOG: PRINT(Addr, Type, Size)
        return cpu_memory(Addr, Type, Size)  

    else:
        if LOG: PRINT('Type', Type)
        if len(Size) == 1:   
            results = [None] * Size[0]  
            for i in range(Size[0]):
                results[i] = cpu_(para[i])
            return results;

        elif len(Size) == 2:
            results = [[None] * Size[1] for _ in range(Size[0])]
            for i in range(Size[0]):
                for j in range(Size[1]):
                    results[i][j] = cpu_(para[i][j])
            return results; 

def cpu_stdvector(para, Type=None, Size=None):
    ftype = para.type.strip_typedefs()

    start = para['_M_impl']['_M_start']
    finish = para['_M_impl']['_M_finish']
    # libstdc++ specializes vector<bool> with bit iterators, not bool pointers.
    if ftype.template_argument(0).strip_typedefs().unqualified().code == gdb.TYPE_CODE_BOOL:
        words = start['_M_p']
        bits = words.type.target().sizeof * 8
        offset = int(start['_M_offset'])
        count = int(finish['_M_p'] - words) * bits + int(finish['_M_offset']) - offset
        Size = [0] if count == 0 else normalize_size(count, Size)
        Type = 'bool'
        return cpu_memory(start, Type, Size)

    Size_t = int(finish - start)

    if Size_t == 0:
        if LOG: PRINT('gdb.TYPE_CODE_STRUCT, Size_t', Size_t, 'Size', Size)
        return np.array([])

    if Type is not None and Size is not None: # top priority
        if LOG: PRINT('gdb.TYPE_CODE_STRUCT, start =', start, ',Type =', Type, ',Size =', Size)
        return cpu_memory(start, Type, Size)    
    
    if LOG: PRINT('gdb.TYPE_CODE_STRUCT type', str(ftype.template_argument(0).strip_typedefs()))
    if Type is None:
        s = str(ftype.template_argument(0).strip_typedefs())
        Type = next((k for k, v in TYPE_MAP.items() if s.startswith(k)), None)    

    if LOG: PRINT('gdb.TYPE_CODE_STRUCT, Size_t', Size_t, 'Size >', Size)
    Size = normalize_size(Size_t, Size)
    if LOG: PRINT('gdb.TYPE_CODE_STRUCT, start', start, 'Type', Type, 'Size', Size)

    if Type is not None:                
        return cpu_memory(start, Type, Size)  

    results = []
    # Decode the complete element, including GLM's anonymous unions, through
    # cpu_ rather than iterating its fields here. Keep a list of records so
    # callers can access position[0], position[1], etc.
    count = math.prod(int(n) for n in np.atleast_1d(Size))
    for i in range(count):
        element = start[i]
        element_type = element.type.strip_typedefs().unqualified()
        if element_type.code == gdb.TYPE_CODE_PTR:
            results.append(int(element))
        else:
            results.append(cpu_(element, None, None))

    return results

def cpu_stdarray(para, Type=None, Size=None):
    ftype = para.type.strip_typedefs().unqualified()
        
    Size_t = int(ftype.template_argument(1))
    if Size_t == 0:
        if LOG: PRINT('Size_t', Size_t)
        return np.array([])

    if Type is not None and Size is not None: # top priority
        if LOG: PRINT('top priority', 'Type', Type, 'Size', Size)
        return cpu_(para['_M_elems'], Type, Size)

    if Type is None:
        Type = str(ftype.template_argument(0).strip_typedefs().unqualified())
        # if LOG: PRINT('para Type', Type)

    Size = normalize_size(Size_t, Size)  

    if Type in TYPE_MAP:
        if LOG: PRINT(f'{Type} in TYPE_MAP')
        Addr = int(para['_M_elems'].address)
        return cpu_memory(Addr, Type, Size)
        
    else:
        # if LOG: PRINT(f'{Type} not in TYPE_MAP')
        results = []
        count = int(np.prod(Size))

        for i in range(count):
            # if LOG: PRINT(f'cpu_stdarray, i = {i}, para = {para["_M_elems"][i]}')
            results.append(cpu_(para['_M_elems'][i], None, None))

        data = np.asarray(results)

        return data.reshape(tuple(Size) + data.shape[1:])

def cpu_stdpair(para):
    return {
        'first': cpu_(para['first'], None, None),
        'second': cpu_(para['second'], None, None),
    }

def cpu_struct(para, Type=None, Size=None):
    if Type is not None and Size is not None:
        return cpu_memory(int(para.address), Type, Size)

    ftype = para.type.strip_typedefs()

    results = {}
    for field in ftype.fields():
        ftype2 = field.type.strip_typedefs().unqualified()
        # GLM vector components live in anonymous unions. GDB exposes those
        # fields without a name; access their named members through the parent.
        if field.name is None:
            if ftype2.code in {gdb.TYPE_CODE_STRUCT, gdb.TYPE_CODE_UNION}:
                for member in ftype2.fields():
                    if member.name is not None:
                        results[member.name] = cpu_(para[member.name], None, None)
                        # Union members alias the same storage; keep the first
                        # name only (GLM uses x/r/s, y/g/t, and z/b/p).
                        if ftype2.code == gdb.TYPE_CODE_UNION:
                            break
            continue
        para2 = para[field.name]

        if ftype2.code in {
                gdb.TYPE_CODE_INT,
                gdb.TYPE_CODE_FLT,
                gdb.TYPE_CODE_BOOL,
                gdb.TYPE_CODE_ENUM,
                gdb.TYPE_CODE_ARRAY,
                gdb.TYPE_CODE_STRUCT,
                gdb.TYPE_CODE_UNION}:
            results[field.name] = cpu_(para2, None, None)

        elif ftype2.code == gdb.TYPE_CODE_PTR:
            results[field.name] = int(para2)

    return results

def cpu_glm_matrix(para, Type=None, Size=None):
    if Type is not None and Size is not None:
        return cpu_memory(int(para.address), Type, Size).transpose()

    ftype = para.type.strip_typedefs().unqualified()
    columns = int(ftype.template_argument(0))
    rows = int(ftype.template_argument(1))
    # Read each column separately so aligned GLM vectors may include padding.
    data = np.asarray([
        [cpu_(para['value'][column][component])
         for component in 'xyzw'[:rows]]
        for column in range(columns)
    ]).transpose()
    if Size is not None:
        shape = tuple(int(n) for n in normalize_size(data.shape, Size))
        return data.reshape(-1)[:math.prod(shape)].reshape(shape)
    return data


def cpu_Eigen_Matrix(para, Type, Size):
    ftype = para.type.strip_typedefs()
 
    while ftype.code in (gdb.TYPE_CODE_REF, gdb.TYPE_CODE_RVALUE_REF):
        para = para.referenced_value()
        ftype = para.type.strip_typedefs()

    ftype = ftype.unqualified() # remove const volatile

    d = para['m_storage']['m_data']
    rows = int(ftype.template_argument(1))
    cols = int(ftype.template_argument(2))

    if rows == -1:
        rows = int(para['m_storage']['m_rows'])
    if cols == -1:
        cols = int(para['m_storage']['m_cols'])

    if LOG: PRINT('rows', rows, 'cols', cols, 'd', d) 

    if d.type.strip_typedefs().code == gdb.TYPE_CODE_PTR:
        if LOG: PRINT('Heap storage int(d)')
        Addr = int(d)                 
    else:
        if LOG: PRINT('Inline storage int(d[array].address)')
        Addr = int(d['array'].address)

    if Type is None:
        Type = str(ftype.template_argument(0).strip_typedefs().unqualified())
        if LOG: PRINT('Type from Matrix', Type)

    Size = normalize_size([rows, cols], Size)

    if(len(Size) == 2):
        SizeT = [Size[1], Size[0]]
        data = cpu_memory(Addr, Type, SizeT)
        data = data.transpose()
        return data

    return cpu_memory(Addr, Type, Size)

def cpu_(para, Type=None, Size=None):
    ftype = para.type.strip_typedefs()
 
    while ftype.code in (gdb.TYPE_CODE_REF, gdb.TYPE_CODE_RVALUE_REF):
        para = para.referenced_value()
        ftype = para.type.strip_typedefs()

    ftype = ftype.unqualified() # remove const volatile

    # if LOG: PRINT('str(ftype)', str(ftype), 'ftype.code', ftype.code, 'Type', Type, 'Size', Size)

    if ftype.code == gdb.TYPE_CODE_PTR: # 1
        if LOG: PRINT('str(ftype)', str(ftype))
    
        if int(para) == 0:
            return None

        target_type = ftype.target().strip_typedefs().unqualified()
        if target_type.code == gdb.TYPE_CODE_STRUCT:
            if LOG: PRINT('ftype.code == gdb.TYPE_CODE_PTR, target_type.code == gdb.TYPE_CODE_STRUCT ')
            if Type is not None and Size is not None:
                return cpu_memory(int(para), Type, Size)

            if Size is None:
                return cpu_(para.dereference(), Type)

            Size_t = math.prod(int(n) for n in np.atleast_1d(Size))
            results = []
            for i in range(Size_t):
                result = cpu_((para + i).dereference(), Type)
                results.append(result)
            return results

        Addr = int(para)
        if Type is None:
            Type = str(target_type)

        return cpu_memory(Addr, Type, Size)

    elif ftype.code == gdb.TYPE_CODE_ARRAY: # 2
        return cpu_array(para, Type, Size)

    elif ftype.code in {gdb.TYPE_CODE_STRUCT, gdb.TYPE_CODE_UNION}: # 3, 4
        # Match only the outer type, excluding template arguments.
        type_name = str(ftype).split('<', 1)[0].strip()
        if type_name == 'std::vector':
            return cpu_stdvector(para, Type, Size)

        elif type_name == 'std::array':
            return cpu_stdarray(para, Type, Size)

        elif type_name == 'std::pair':
            return cpu_stdpair(para)

        elif type_name in {'std::filesystem::path', 'std::filesystem::__cxx11::path'}:
            return cpu_(para['_M_pathname'])

        elif type_name in {'std::basic_string', 'std::__cxx11::basic_string'}:
            n = int(para['_M_string_length'])
            s = para['_M_dataplus']['_M_p'].string(length=n)
            return np.str_(s)  

        elif type_name == 'Eigen::Matrix':
            return cpu_Eigen_Matrix(para, Type, Size)

        elif type_name == 'glm::mat':
            return cpu_glm_matrix(para, Type, Size)

        else:
            return cpu_struct(para, Type, Size)
    
    elif ftype.code == gdb.TYPE_CODE_ENUM: # 5
        if ftype.sizeof == 1:
            return np.int8(para) if ftype.is_signed else np.uint8(para)
        elif ftype.sizeof == 2:
            return np.int16(para) if ftype.is_signed else np.uint16(para)
        elif ftype.sizeof == 4:
            return np.int32(para) if ftype.is_signed else np.uint32(para)
        elif ftype.sizeof == 8:
            return np.int64(para) if ftype.is_signed else np.uint64(para)
        else:
            return value

    elif ftype.code == gdb.TYPE_CODE_INT: # 8
        if LOG: PRINT('ftype.code', gdb.TYPE_CODE_INT, 'ftype.sizeof', ftype.sizeof)
        if ftype.sizeof == 1:
            return np.int8(para) if ftype.is_signed else np.uint8(para)
        elif ftype.sizeof == 2:
            return np.int16(para) if ftype.is_signed else np.uint16(para)
        elif ftype.sizeof == 4:
            return np.int32(para) if ftype.is_signed else np.uint32(para)
        else:
            return np.int64(para) if ftype.is_signed else np.uint64(para)

    elif ftype.code == gdb.TYPE_CODE_FLT: # 9
        if ftype.sizeof == 4:
            return np.float32(para)
        elif ftype.sizeof == 8:
            return np.float64(para)
        else:
            return np.longdouble(para)

    elif ftype.code == gdb.TYPE_CODE_BOOL: # 21
        return np.uint8(int(para) != 0)

def cpu(Name, Type=None, Size=None):
    if LOG: PRINT('Name', Name, 'Type', Type, 'Size', Size)
    para = gdb.parse_and_eval(Name)
    return cpu_(para, Type, Size)


def gpu_struct(para, Type=None, Size=None):
    count = 1 if Size is None else math.prod(np.atleast_1d(Size))
    results = []
    for i in range(count):
        struct = para[i]
        ftype = struct.type.strip_typedefs().unqualified()

        result = {}

        for field in ftype.fields():
            ftype2 = field.type.strip_typedefs().unqualified()
            para2 = struct[field.name]      

            if ftype2.code in {
                    gdb.TYPE_CODE_INT,
                    gdb.TYPE_CODE_FLT,
                    gdb.TYPE_CODE_BOOL,
                    gdb.TYPE_CODE_ENUM,
                    gdb.TYPE_CODE_ARRAY,
                    gdb.TYPE_CODE_STRUCT}:
                result[field.name] = cpu_(para2, None, None)

            elif ftype2.code == gdb.TYPE_CODE_PTR:
                result[field.name] = int(para2)
    
        results.append(result)

    return results

def gpu_buffer_size(d_addr):
    base_addr = 0
    size_addr = 0
    try:
        base_addr = int(gdb.parse_and_eval(
            '((unsigned long long*)calloc(1, sizeof(unsigned long long)))'
        ))
        size_addr = int(gdb.parse_and_eval(
            '((size_t*)calloc(1, sizeof(size_t)))'
        ))

        expr = (
            f'((int (*)(unsigned long long*, size_t*, unsigned long long))cuMemGetAddressRange_v2)('
            f'(unsigned long long*){base_addr}, '
            f'(size_t*){size_addr}, '
            f'(unsigned long long){d_addr})'
        )
        ret = int(gdb.parse_and_eval(expr))

        base = int(gdb.parse_and_eval(f'*(unsigned long long*){base_addr}'))
        size = int(gdb.parse_and_eval(f'*(size_t*){size_addr}'))

        return size
        
    finally:
        if size_addr:
            gdb.parse_and_eval(f'((void (*)(void*))free)((void*){size_addr})')
        if base_addr:
            gdb.parse_and_eval(f'((void (*)(void*))free)((void*){base_addr})')

def gpu_(para, Type=None, Size=None):
    # return cpu_(para, Type, Size)
    ftype = para.type.strip_typedefs()

    while ftype.code in (gdb.TYPE_CODE_REF, gdb.TYPE_CODE_RVALUE_REF):
        para = para.referenced_value()
        ftype = para.type.strip_typedefs()  

    ftype = ftype.unqualified() # remove const volatile

    if ftype.code == gdb.TYPE_CODE_PTR: # 1
        if LOG: PRINT('str(ftype)', str(ftype))
    
        if int(para) == 0:
            return None

        target_type = ftype.target().strip_typedefs().unqualified()
        if target_type.code == gdb.TYPE_CODE_STRUCT:
            if LOG: PRINT('ftype.code == gdb.TYPE_CODE_PTR, target_type.code == gdb.TYPE_CODE_STRUCT ')
            Size = [math.prod(Size)]
            return gpu_struct(para, Type, Size)

        Addr = int(para)
        if Type is None:
            Type = str(target_type)

        return cpu_memory(Addr, Type, Size)
  
    elif ftype.code == gdb.TYPE_CODE_INT: # 8
        if LOG: PRINT('ftype.code', gdb.TYPE_CODE_INT, 'ftype.sizeof', ftype.sizeof)
        if ftype.sizeof == 1:
            return np.int8(para) if ftype.is_signed else np.uint8(para)
        elif ftype.sizeof == 2:
            return np.int16(para) if ftype.is_signed else np.uint16(para)
        elif ftype.sizeof == 4:
            return np.int32(para) if ftype.is_signed else np.uint32(para)
        else:
            return np.int64(para) if ftype.is_signed else np.uint64(para)

    elif ftype.code == gdb.TYPE_CODE_FLT: # 9
        if LOG: PRINT('ftype.code', gdb.TYPE_CODE_INT, 'ftype.sizeof', ftype.sizeof)
        if ftype.sizeof == 4:
            return np.float32(para)
        elif ftype.sizeof == 8:
            return np.float64(para)
        else:
            return np.longdouble(para)

    else:
        if LOG: PRINT(f'{name}({ftype.code}): fail')

    return None

def gpu(Name, Type=None, Size=None):
    if LOG: PRINT('Name', Name, 'Type', Type, 'Size', Size)
    para = gdb.parse_and_eval(Name)

    if para.is_optimized_out:
        print(f"gpu: {Name!r} is optimized out in the selected frame/thread; "
              "pause after its assignment where the value is still live.",
              file=sys.stderr)
        return None

    ftype = para.type.strip_typedefs()

    while ftype.code in (gdb.TYPE_CODE_REF, gdb.TYPE_CODE_RVALUE_REF):
        para = para.referenced_value()
        ftype = para.type.strip_typedefs()  

    ftype = ftype.unqualified() # remove const volatile

    if ftype.code != gdb.TYPE_CODE_PTR: # 1
        if LOG: PRINT('ftype.code != gdb.TYPE_CODE_PTR', ftype.code)
        # Kernel locals are values, not CUDA allocation addresses. Read them
        # directly without calling host allocation or CUDA copy functions.
        return gpu_(para, Type, Size)

    if LOG: PRINT('ftype.code == gdb.TYPE_CODE_PTR', ftype.code)

    if Type is not None:
        target = gdb.lookup_type(Type)
        if LOG: PRINT('Type', Type, '=> str(target)', str(target))
    else:
        target = ftype.target().strip_typedefs().unqualified()
        if LOG: PRINT('str(ftype)', str(ftype), '=> str(target)', str(target))
    
    d_addr = int(para)

    d_addr_size = gpu_buffer_size(d_addr)
    Size_new = int(d_addr_size / target.sizeof)

    if LOG: PRINT('d_addr_size', d_addr_size, 'Size_new', Size_new, 'Size', Size)
    Size = normalize_size(Size_new, Size)

    byte_count = target.sizeof * math.prod(Size)  

    # expr = f'((unsigned char*)malloc({byte_count}))'
    expr = f'((unsigned char*)calloc(1, {byte_count}))'
    if LOG: PRINT('expr', expr, [target.sizeof, math.prod(Size)])

    h_addr = int(gdb.parse_and_eval(expr))
    
    try:
        if LOG: PRINT('h_addr', h_addr, 'd_addr', d_addr, 'byte_count', byte_count)

        expr = f'((int (*)(void*, const void*, size_t, int))cudaMemcpy)((void*){h_addr}, (void*){d_addr}, {byte_count}, cudaMemcpyDeviceToHost)'
        if LOG: PRINT('expr', expr)
        ret = gdb.parse_and_eval(expr)

        para2 = gdb.Value(h_addr).cast(target.pointer())
        data = gpu_(para2, Type, Size)

        return data

    except Exception as e:
        if LOG: PRINT(f"Exception {type(e).__name__}: {e}")
        return None

    finally:
        expr = f'((void(*)(void*))free)((void*){h_addr})'
        if LOG: PRINT('expr', expr)
        gdb.parse_and_eval(expr)

def kernel_memory(para, Type, Size, Step):
    if LOG: PRINT('para', para, 'Type', Type, 'Size', Size, 'Step', Step)

    if Size is not None and Step == 1 and len(Size) > 1:
        if LOG: PRINT('No step operation')

        type_t = TYPE_MAP[Type]
        data_size = np.dtype(type_t).itemsize
        data_size *= int(np.prod(Size))

        if LOG: PRINT('type_t', type_t, 'data_size', data_size, [np.dtype(type_t).itemsize, int(np.prod(Size))])

        if LOG: PRINT('para', para, 'para.dereference()', para.dereference(), 'for Test')

        total_size = data_size
        if data_size % 8 == 0:
            total_size = int(data_size / 8)
            total_data = np.empty(total_size, dtype=np.ulonglong)
            total_type = gdb.lookup_type("unsigned long long")
        elif data_size % 4 == 0:
            total_size = int(data_size / 4)
            total_data = np.empty(total_size, dtype=np.uint32)
            total_type = gdb.lookup_type("unsigned int")
        elif data_size % 2 == 0:
            total_size = int(data_size / 2)
            total_data = np.empty(total_size, dtype=np.uint16)
            total_type = gdb.lookup_type("unsigned short")    
        else:
            total_size = int(data_size)
            total_data = np.empty(total_size, dtype=np.uint8)
            total_type = gdb.lookup_type("unsigned char")  

        for i in range(total_size):
            total_offset = int(i * total_data.itemsize / para.type.target().sizeof)
            total_data[i] = (para + total_offset).cast(total_type.pointer()).dereference()

        data = total_data.view(type_t)
        return data

    elif Size is not None and Step > 1 and len(Size) > 1:
        if LOG: PRINT('Step operation')

        type_t = TYPE_MAP[Type]
        data = np.empty(Size, dtype=type_t)

        for i in range(0, Size[0], Step):
            for j in range(0, Size[1], Step):
                data[i, j] = (para + i * Size[1] + j).dereference()

                for di in range(Step): 
                    ii = i + di
                    if ii >= shape[0]:
                        continue

                    for dj in range(Step):
                        jj = j + dj
                        if jj >= Size[1]:
                            continue
                        
                        data[ii, jj] = data[i, j];
        return data
    elif Size is not None and len(Size) == 1:
        count = int(Size[0])
        if count != Size[0] or count < 0:
            raise ValueError('Size must be a nonnegative integer')
        if int(Step) != Step or Step < 1:
            raise ValueError('Step must be a positive integer')
        Step = int(Step)
        dtype = np.dtype(TYPE_MAP[Type])
        data = np.empty(count, dtype=dtype)
        for i in range(0, count, Step):
            value = (para + i).dereference()
            value = float(value) if dtype.kind == 'f' else int(value)
            data[i:min(i + Step, count)] = value
        return data
    elif Size is None:
        # Probe a bounded readable prefix using the original CUDA pointer.
        # Readability does not identify an allocation's exact boundary.
        limit = 1024
        values = []
        if int(Step) != Step or Step < 1:
            raise ValueError('Step must be a positive integer')
        Step = int(Step)
        dtype = np.dtype(TYPE_MAP[Type])
        for Size_t in range(limit):
            try:
                value = (para + Size_t).dereference()
                # Force lazy GDB values to be read inside the try block.
                value = float(value) if dtype.kind == 'f' else int(value)
            except gdb.error:
                if not values:
                    raise
                break
            values.append(value)
        Size_t = len(values)
        data = np.asarray(values, dtype=dtype)
        if Step > 1:
            data = data[(np.arange(Size_t) // Step) * Step]
        return data

def kernel_array(para, Type=None, Size=None):
    ftype = para.type.strip_typedefs().unqualified()
    dims = []
    element_type = ftype
    while element_type.code == gdb.TYPE_CODE_ARRAY:
        lo, hi = element_type.range()
        dims.append(hi - lo + 1)
        element_type = element_type.target().strip_typedefs().unqualified()

    shape = normalize_size(dims, Size)
    count = math.prod(shape)
    results = []

    def read_elements(value):
        value_type = value.type.strip_typedefs().unqualified()
        if value_type.code == gdb.TYPE_CODE_ARRAY:
            lo, hi = value_type.range()
            for i in range(lo, hi + 1):
                if len(results) >= count:
                    break
                # Index the original value to preserve CUDA's @local address space.
                read_elements(value[i])
        elif value_type.code == gdb.TYPE_CODE_PTR:
            results.append(int(value))
        else:
            results.append(cpu_(value, None, None))

    read_elements(para)
    dtype = TYPE_MAP[Type] if Type is not None else None
    return np.asarray(results, dtype=dtype).reshape(shape)

def kernel_struct(para, Type=None, Size=None):
    ftype = para.type.strip_typedefs()

    results = {}
    for field in ftype.fields():
        ftype2 = field.type.strip_typedefs().unqualified()
        para2 = para[field.name]      

        if ftype2.code in {
                gdb.TYPE_CODE_INT,
                gdb.TYPE_CODE_FLT,
                gdb.TYPE_CODE_BOOL,
                gdb.TYPE_CODE_ENUM,
                gdb.TYPE_CODE_ARRAY,
                gdb.TYPE_CODE_STRUCT}:
            results[field.name] = kernel_(para2, None, None)

        elif ftype2.code == gdb.TYPE_CODE_PTR:
            results[field.name] = int(para2)

    return results

def kernel_(para, Type=None, Size=None, Step = 1):  
    ftype = para.type.strip_typedefs()

    while ftype.code in (gdb.TYPE_CODE_REF, gdb.TYPE_CODE_RVALUE_REF):
        para = para.referenced_value()
        ftype = para.type.strip_typedefs()  

    ftype = ftype.unqualified() # remove const volatile

    if LOG: PRINT('ftype', str(ftype), 'ftype.code', ftype.code)

    if ftype.code == gdb.TYPE_CODE_PTR: # 1
        if LOG: PRINT('str(ftype)', str(ftype))
    
        if int(para) == 0:
            return None

        target_type = ftype.target().strip_typedefs().unqualified()
        if target_type.code == gdb.TYPE_CODE_STRUCT:
            if LOG: PRINT('ftype.code == gdb.TYPE_CODE_PTR, target_type.code == gdb.TYPE_CODE_STRUCT ')
            return kernel_struct(para.dereference(), Type, Size)

        Addr = int(para) # Normal PTR
        if Type is None:
            Type = str(target_type).replace('@generic ', '')

        if LOG: PRINT('Addr', Addr, 'Type', Type, 'Size', Size, 'Step', Step)
        return kernel_memory(para, Type, Size, Step)
  
    elif ftype.code == gdb.TYPE_CODE_ARRAY: # 2
        return kernel_array(para, Type, Size)  

    if ftype.code == gdb.TYPE_CODE_STRUCT:
        return kernel_struct(para, Type, Size)

    elif ftype.code == gdb.TYPE_CODE_INT: # 8
        if LOG: PRINT('ftype.code', gdb.TYPE_CODE_INT, 'ftype.sizeof', ftype.sizeof)
        if ftype.sizeof == 1:
            return np.int8(para) if ftype.is_signed else np.uint8(para)
        elif ftype.sizeof == 2:
            return np.int16(para) if ftype.is_signed else np.uint16(para)
        elif ftype.sizeof == 4:
            return np.int32(para) if ftype.is_signed else np.uint32(para)
        else:
            return np.int64(para) if ftype.is_signed else np.uint64(para)

    elif ftype.code == gdb.TYPE_CODE_FLT: # 9
        if LOG: PRINT('ftype.code', gdb.TYPE_CODE_INT, 'ftype.sizeof', ftype.sizeof)
        if ftype.sizeof == 4:
            return np.float32(para)
        elif ftype.sizeof == 8:
            return np.float64(para)
        else:
            return np.longdouble(para)

    else:
        if LOG: PRINT(f'{name}({ftype.code}): fail')

    return None

def kernel(Name, Type=None, Size=None, Step = 1):  
    if LOG: PRINT('Name', Name, 'Type', Type, 'Size', Size)
    para = gdb.parse_and_eval(Name)
    return kernel_(para, Type, Size, Step)

def is_device_pointer(Name):
    para = gdb.parse_and_eval(Name)
    ftype = para.type.strip_typedefs().unqualified()

    # Scalars and aggregates are values, not allocation addresses. In a CUDA
    # frame, attempting a host function call here can lose the device focus.
    if ftype.code != gdb.TYPE_CODE_PTR:
        return 2 if '@' in str(ftype) else 0

    if '@' not in str(ftype):
        sym, is_field  = gdb.lookup_symbol("cudaPointerGetAttributes")
        if LOG: PRINT('sym', sym)

        try:
            buf = int(gdb.parse_and_eval("(void *) malloc(64)"))

            if LOG: PRINT(f'(int) cudaPointerGetAttributes((void *) {buf:#x}, {Name})')
            rc  = int(gdb.parse_and_eval(f'(int) cudaPointerGetAttributes((void *) {buf:#x}, {Name})'))

            attr   = int(gdb.parse_and_eval(f'*(int *) {buf:#x}'))

            if LOG: PRINT(f'rc = {rc}, type = {attr}')
            if LOG: PRINT({0: 'host-malloc', 1: 'pinned', 2: 'device', 3: 'managed'}.get(attr, '?'))

            gdb.parse_and_eval(f'(void) free((void *) {buf:#x})')

            if attr in [2, 3]:
                if LOG: PRINT('is_device_pointer success:', 'mem kernel host 1')
                return 1
            else:
                if LOG: PRINT('is_device_pointer success:', 'mem cpu 0')
                return 0
        
        except gdb.error as e:
            if LOG: PRINT('is_device_pointer fail:', e, 'mem cpu 0')
            return 0
    else:
        if LOG: PRINT('is_device_pointer success:', 'mem kernel kernel 2')
        return 2

def mem_(Name, Type=None, Size=None, Step=1):
    if isinstance(Type, str) and Type == '':
        Type = None
    if isinstance(Size, str):
        if Size == '': Size = None
        else: Size = mem(Size)

    expression = Name.strip().rstrip(';').rstrip()
    if '(' in expression:
        return int(gdb.parse_and_eval(expression))

    TYPE = {
        'uc':        'unsigned char',
        'c':         'char',
        'us':        'unsigned short',
        's':         'short',
        'ui':        'unsigned int',
        'i':         'int',
        'ull':       'unsigned long long',
        'ul':        'unsigned long',
        'f':         'float',
        'd':         'double',
    }
    Type = next((v for k, v in TYPE.items() if k == Type), Type)

    Size = np.atleast_1d(np.asarray(Size)) if Size is not None else None

    if LOG: PRINT('Name', Name, 'Type', Type, 'Size', Size, 'Step', Step)

    is_device = is_device_pointer(Name)
    if LOG: PRINT('Name', Name, 'is device', is_device)

    if is_device == 0:
        return cpu(Name, Type, Size)

    elif is_device == 1:
        return gpu(Name, Type, Size)

    elif is_device == 2:
        return kernel(Name, Type, Size, Step)
  
    return None

def aware(console = 0):
    try:
        kernels_output = gdb.execute('info cuda kernels', to_string=True)
    except gdb.error:
        return False

    kernels_match = re.search(
        r'^\s*\*[^\n]*?(\(\d+\s*,\s*\d+\s*,\s*\d+\))\s+'
        r'(\(\d+\s*,\s*\d+\s*,\s*\d+\))',
        kernels_output, re.MULTILINE)

    BLOCK, KERNEL = (
        [np.array([int(value) for value in group.strip('()').split(',')])
         for group in kernels_match.groups()]
        if kernels_match else (np.zeros(3, dtype=int), np.zeros(3, dtype=int))
    )

    try:
        kernel_output = gdb.execute('cuda block thread', to_string=True)
    except gdb.error:
        return False

    kernel_match = re.search(
        r'(\(\d+\s*,\s*\d+\s*,\s*\d+\)),?\s+(?:thread\s+)?'
        r'(\(\d+\s*,\s*\d+\s*,\s*\d+\))',
        kernel_output, re.MULTILINE)

    block, kernel = (
        [np.array([int(value) for value in group.strip('()').split(',')])
         for group in kernel_match.groups()]
        if kernel_match else (np.zeros(3, dtype=int), np.zeros(3, dtype=int))
    )

    id = block * KERNEL + kernel if kernels_match and kernel_match else None

    if console:
        with np.printoptions(formatter={'int_kind': lambda value: str(value)}):
            print(id, '/', block, kernel, '/', BLOCK, KERNEL, '<-- kernel')

def mem(Name, Type=None, Size=None, Step=1):
    try:
        frame = gdb.selected_frame()
        sym, _ = gdb.lookup_symbol(Name, frame.block())
        if sym == None:
            if LOG: PRINT('sym', sym)
            aware(1)
    except (gdb.error, RuntimeError) as error:
        PRINT(f'{type(error).__name__}: {error}')
        aware(1)

    if LOG:
        data = mem_(Name, Type, Size, Step)
        # np.save(f'{PATH}/data/{Name}.npy', data) 
    else:
        try:
            data = mem_(Name, Type, Size, Step)
            # np.save(f'{PATH}/data/{Name}.npy', data) 
        except Exception as error:
            PRINT(f"mem: cannot read {Name!r}: {error}", file=sys.stderr)
            data = None
            
    return data

class L(np.ndarray):
    def __new__(cls, data):
        return np.asarray(data, dtype=object).view(cls)

    def __getitem__(self, key):
        if isinstance(key, str):
            fields = key.split('.')
            values = []

            for record in np.asarray(self).flat:
                value = record
                for field in fields:
                    value = value[field]
                values.append(value)

            if self.ndim == 0:
                return values[0]
            return np.asarray(values).reshape(self.shape)

        return super().__getitem__(key)

def MEM(Name, Type=None, Size=None, Step=1):
    data = mem_(Name, Type, Size, Step)
    return L(data)

# Value  Constant                     Meaning
# -1     TYPE_CODE_BITSTRING          Bit string (deprecated, kept for compatibility)
# 1      TYPE_CODE_PTR                Pointer, e.g. T*
# 2      TYPE_CODE_ARRAY              Array, e.g. T[N]
# 3      TYPE_CODE_STRUCT             Struct or C++ class
# 4      TYPE_CODE_UNION              Union
# 5      TYPE_CODE_ENUM               Enumeration
# 6      TYPE_CODE_FLAGS              Bit-flags register type (e.g. eflags)
# 7      TYPE_CODE_FUNC               Function type
# 8      TYPE_CODE_INT                Integer (int, long, size_t, ...)
# 9      TYPE_CODE_FLT                Floating point (float, double)
# 10     TYPE_CODE_VOID               void
# 11     TYPE_CODE_SET                Set type (Pascal / Modula-2)
# 12     TYPE_CODE_RANGE              Range type (array index bounds, etc.)
# 13     TYPE_CODE_STRING             String type (Fortran-style, NOT char*)
# 14     TYPE_CODE_ERROR              Unrecognized / corrupt debug info
# 15     TYPE_CODE_METHOD             C++ member function type
# 16     TYPE_CODE_METHODPTR          Pointer to member function, void (T::*)()
# 17     TYPE_CODE_MEMBERPTR          Pointer to data member, int T::*
# 18     TYPE_CODE_REF                C++ lvalue reference, T&
# 19     TYPE_CODE_RVALUE_REF         C++ rvalue reference, T&&
# 20     TYPE_CODE_CHAR               Character type, char
# 21     TYPE_CODE_BOOL               Boolean, bool
# 22     TYPE_CODE_COMPLEX            Complex number, _Complex
# 23     TYPE_CODE_TYPEDEF            typedef / using alias (not yet stripped)
# 24     TYPE_CODE_NAMESPACE          C++ namespace
# 25     TYPE_CODE_DECFLOAT           Decimal float (_Decimal64, etc.)
# 26     TYPE_CODE_MODULE             Module (Fortran)
# 27     TYPE_CODE_INTERNAL_FUNCTION  GDB internal convenience function ($_strlen, ...)
# 28     TYPE_CODE_XMETHOD            Python-defined extension method (xmethod)
# 29     TYPE_CODE_FIXED_POINT        Fixed-point number type
# 30     TYPE_CODE_NAMELIST           Fortran namelist

# python3 -m mem mem.py
# sudo du -xhd3 / | sort -h
# gio trash --empty

# node --version
# npm --version
# sudo npm install -g @vscode/vsce
# ~/.vscode/extensions
# ~/.config/Code/User/keybindings.json
# Ctrl + Shift + P
# Reload Window
# Extension Development Host
# Extensions: Install from VSIX...

# Credit balance is too low
# unset ANTHROPIC_API_KEY
# echo "[$ANTHROPIC_API_KEY]"
# /login "Claude account with subscription"
# /status # /usage
