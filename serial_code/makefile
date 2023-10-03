# Folder for the .o files
OBJDIR := obj

# Finds all .cpp files and creates needed .o files
SOURCES := $(shell find . -name '*.cpp' -printf "%f\n")
OBJECTS := $(addprefix $(OBJDIR)/,$(SOURCES:%.cpp=%.o))

# Linking
all: $(OBJECTS)
	g++ $(OBJECTS) -o app_serial

# Compilation to .o
$(OBJDIR)/%.o: %.cpp
	g++ -c $< -o $@

# Cleaning of binaries and .o files
clean:
	rm -f $(OBJDIR)/*.o app_serial