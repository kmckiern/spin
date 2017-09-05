class Electron(object):
    """
    Representation of an electron
    
    Attributes:
        s: spin quantum number
    """
    
    def __init__(self, s=0.5):
        """
        Returns an electron object with a specified spin quantum number, s
        """
        self._s = s
    
    def spin(self):
        """
        Returns the electron spin
        """
        return self._s
    
    def flip(self):
        """
        Flips the electron spin
        """
        self._s *= -1
