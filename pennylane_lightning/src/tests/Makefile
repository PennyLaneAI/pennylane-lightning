# Copyright 2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
CC = g++
CFLAGS = -std=c++11 -g -Wall -fopenmp -fPIC -I/usr/include \
	-I../include -I$(GOOGLETEST_DIR)/include \
	-march=native

LFLAGS = -fopenmp -L$(GOOGLETEST_DIR)/lib -lgtest

all: test

lightning_util_unittest.o: lightning_util_unittest.cpp
	$(CC) $^ $(CFLAGS) -c

Gates.o: ../Gates.cpp
	$(CC) $^ $(CFLAGS) -c

Apply.o: ../Apply.cpp
	$(CC) $^ $(CFLAGS) -c

lightning_gates_unittest.o: lightning_gates_unittest.cpp
	$(CC) $^ $(CFLAGS) -c

lightning_apply_unittest.o: lightning_apply_unittest.cpp
	$(CC) $^ $(CFLAGS) -c

gtest_main.o: gtest_main.cpp
	$(CC) $^ $(CFLAGS) -c

cpptests: gtest_main.o lightning_util_unittest.o lightning_gates_unittest.o Gates.o lightning_apply_unittest.o Apply.o
	$(CC) $^ -o $@ $(LFLAGS)

test: cpptests
	./cpptests

clean:
	rm -rf *~ cpptests *.out *.o *.so *.pyc *.mod

