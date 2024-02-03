## Using the wavl24 wavelet package

D. Candela 2/3/24

## Installing

- Activate desired environment, then clone or download `wavl24` repo, and cd into repo:
  
  ```
  (base) somewhere$ conda activate myenv
  # This shows clone, alternatively download ZIP of files and extract:
  (myenv) somewhere$ git clone <HTTP or SSH address of wavl24 repo>
  (myenv) somewhere$ cd wavl24
  ```

- Do editable install of walv24 package:
  
  ```
  (myenv) somewhere/wavl24 pip install -e .
  ```

- Now can import and use `wavl` module:
  
  ```
  import numpy as np
  import matplotlib.pyplot as plt
  from wavl24 import wavl
  
  if __name__=='__main__':
      # Plot like NR3 Fig. 13.10.1 top.
      a = np.zeros(1024)
      a[4] = 1.0
      wavl.wt1(a,wavl.Daub4,forward=False)
      plt.plot(a)
      plt.show()
  ```


