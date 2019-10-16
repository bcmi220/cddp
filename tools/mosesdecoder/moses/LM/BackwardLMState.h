// $Id$

/***********************************************************************
Moses - factored phrase-based language decoder
Copyright (C) 2006 University of Edinburgh

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
***********************************************************************/

#ifndef moses_BackwardLMState_h
#define moses_BackwardLMState_h

#include "moses/FF/FFState.h"
#include "moses/LM/Backward.h"

#include "lm/state.hh"
/*
namespace lm {
  namespace ngram {
    class ChartState;
  }
}
*/

//#include "lm/state.hh"

namespace Moses
{

//template<typename M>
class BackwardLanguageModelTest;

class BackwardLMState : public FFState
{

public:

  size_t hash() const;
  virtual bool operator==(const FFState& other) const;

  // Allow BackwardLanguageModel to access the private members of this class
  template <class Model> friend class BackwardLanguageModel;

  //    template <class Model> friend class Moses::BackwardLanguageModelTest;
  friend class Moses::BackwardLanguageModelTest;

private:
  lm::ngram::ChartState state;

};

}

#endif
