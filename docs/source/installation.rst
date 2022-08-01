Installation Guide
==================

Normal Bingo
------------

To install Bingo, simply use pip:

.. code-block:: console

    pip install bingo-nasa


Strongly-Typed Bingo
--------------------

To use Bingo's strongly-typed features, you will have to install Bingo's
source code and checkout the ``stronglyTyped`` branch:

.. code-block:: console

    git clone https://github.com/nightdr/bingo.git
    cd bingo
    git checkout stronglyTyped

and make sure you have the requirements necessary to use Bingo (mpi4py optional):

.. code-block:: console

    pip install -r requirements.txt
