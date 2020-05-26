class MaskFn():
  def mask_types(self):
    raise NotImplementedError()

  @classmethod
  def mask_type_serialize(cls, m_type):
    raise NotImplementedError()

  @classmethod
  def mask(cls, doc):
    raise NotImplementedError()
