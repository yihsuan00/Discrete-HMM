.PHONY: all clean

CFLAGS+=
LDFLAGS+=-lm     # link to math library

TARGET=train test accuracy

all: $(TARGET)
train:train.cpp hmm.h
	g++ train.cpp -o train
test:test.cpp hmm.h
	g++ test.cpp -o tests
# type make/make all to compile test_hmm

clean:
	$(RM) $(TARGET)   # type make clean to remove the compiled file
