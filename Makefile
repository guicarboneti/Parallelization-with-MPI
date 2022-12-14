NAME		=       nbody
CC		=	mpicc
CXX		=	icpc
LD		=	mpicc
DEBUG		+=
CFLAGS		+=	-O3 $(DEBUG)
CXXFLAGS	+=	$(CFLAGS)
LDFLAGS		+=	-lm

EXE		=	$(NAME)
CFILES		=	$(wildcard *.c)
CXXFILES	=	$(wildcard *.cpp)
OBJECTS		=	$(CFILES:.c=.o) $(CXXFILES:.cpp=.o)

all : $(EXE)
	@echo "Building..."

$(EXE) : $(OBJECTS)
	$(LD) $^ $(CFLAGS) $(LDFLAGS) -o $@

%.o : %.c
	$(CC) $(CFLAGS) $(LDFLAGS) -c $< -o $@

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean :	
	rm -fr $(EXE) $(OBJECTS) 
