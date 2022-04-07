###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda-11.2

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=/usr/bin/gcc
CC_FLAGS= -Wextra -O3 -fomit-frame-pointer -march=native -mavx2
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=-arch sm_75
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src
SVM_DIR = svm

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

##########################################################

## Make variables ##

# Target executable name:
EXE = svm_ipfe-gpu

# Object files:
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/rlwe_sife_gpu.o $(OBJ_DIR)/crt_gpu.o $(OBJ_DIR)/ntt_gpu.o $(OBJ_DIR)/sample_gpu.o  $(OBJ_DIR)/arith_rns.o $(OBJ_DIR)/crt.o $(OBJ_DIR)/randombytes.o $(OBJ_DIR)/gauss.o $(OBJ_DIR)/ntt.o $(OBJ_DIR)/function.o $(OBJ_DIR)/rlwe_sife.o $(OBJ_DIR)/sample.o $(OBJ_DIR)/aes256ctr.o $(OBJ_DIR)/svm.o 

##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) -lgmp

# Compile main.c file to object files:
$(OBJ_DIR)/%.o : %.c
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CC_FLAGS) -c $< -o $@ -lgmp

# Compile svm source files to object files:
$(OBJ_DIR)/%.o : $(SVM_DIR)/%.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@ -lgmp

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : src/%.cu 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)




