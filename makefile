# Folders for the source and .o files
SOURCEDIR := src
OBJDIR := obj

# Compiler and needed flags
CC := nvcc
CFLAGS := -arch=native

# Finds all .cu files in source folder and creates needed .o files
SOURCES := $(shell find $(SOURCEDIR) -name '*.cu' -printf "%f\n")
OBJECTS := $(addprefix $(OBJDIR)/,$(SOURCES:%.cu=%.o))

# Linking
all: $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o app

# Compilation to .o
$(OBJDIR)/%.o: $(SOURCEDIR)/%.cu
	$(CC) $(CFLAGS) -dc $< -o $@

# Cleaning of binaries and .o files
clean:
	rm -f $(OBJDIR)/*.o app