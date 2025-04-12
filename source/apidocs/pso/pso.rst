:py:mod:`pso`
=============

.. py:module:: pso

.. autodoc2-docstring:: pso
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`cardinate <pso.cardinate>`
     - .. autodoc2-docstring:: pso.cardinate
          :summary:
   * - :py:obj:`City <pso.City>`
     - .. autodoc2-docstring:: pso.City
          :summary:
   * - :py:obj:`Particle <pso.Particle>`
     - .. autodoc2-docstring:: pso.Particle
          :summary:
   * - :py:obj:`PSO <pso.PSO>`
     - .. autodoc2-docstring:: pso.PSO
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`update_best_solutions <pso.update_best_solutions>`
     - .. autodoc2-docstring:: pso.update_best_solutions
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`dist <pso.dist>`
     - .. autodoc2-docstring:: pso.dist
          :summary:
   * - :py:obj:`MaxFES <pso.MaxFES>`
     - .. autodoc2-docstring:: pso.MaxFES
          :summary:
   * - :py:obj:`FES <pso.FES>`
     - .. autodoc2-docstring:: pso.FES
          :summary:
   * - :py:obj:`object_input <pso.object_input>`
     - .. autodoc2-docstring:: pso.object_input
          :summary:
   * - :py:obj:`mstsp_pr <pso.mstsp_pr>`
     - .. autodoc2-docstring:: pso.mstsp_pr
          :summary:

API
~~~

.. py:data:: dist
   :canonical: pso.dist
   :value: 'array(...)'

   .. autodoc2-docstring:: pso.dist

.. py:data:: MaxFES
   :canonical: pso.MaxFES
   :value: 'array(...)'

   .. autodoc2-docstring:: pso.MaxFES

.. py:data:: FES
   :canonical: pso.FES
   :value: 0

   .. autodoc2-docstring:: pso.FES

.. py:data:: object_input
   :canonical: pso.object_input
   :value: 0

   .. autodoc2-docstring:: pso.object_input

.. py:data:: mstsp_pr
   :canonical: pso.mstsp_pr
   :value: ['simple1_9', 'simple2_10', 'simple3_10', 'simple4_11', 'simple5_12', 'simple6_12', 'geometry1_10', ...

   .. autodoc2-docstring:: pso.mstsp_pr

.. py:class:: cardinate(a=0, b=0)
   :canonical: pso.cardinate

   .. autodoc2-docstring:: pso.cardinate

   .. rubric:: Initialization

   .. autodoc2-docstring:: pso.cardinate.__init__

.. py:class:: City(input_data)
   :canonical: pso.City

   Bases: :py:obj:`object`

   .. autodoc2-docstring:: pso.City

   .. rubric:: Initialization

   .. autodoc2-docstring:: pso.City.__init__

   .. py:attribute:: city_vec
      :canonical: pso.City.city_vec
      :value: 'array(...)'

      .. autodoc2-docstring:: pso.City.city_vec

   .. py:attribute:: random
      :canonical: pso.City.random
      :value: 'array(...)'

      .. autodoc2-docstring:: pso.City.random

   .. py:attribute:: onpath
      :canonical: pso.City.onpath
      :value: 'array(...)'

      .. autodoc2-docstring:: pso.City.onpath

   .. py:attribute:: fitness
      :canonical: pso.City.fitness
      :value: 0

      .. autodoc2-docstring:: pso.City.fitness

   .. py:attribute:: path_length
      :canonical: pso.City.path_length
      :value: 0

      .. autodoc2-docstring:: pso.City.path_length

   .. py:attribute:: d
      :canonical: pso.City.d
      :value: 'cardinate(...)'

      .. autodoc2-docstring:: pso.City.d

   .. py:method:: distance(a, b)
      :canonical: pso.City.distance

      .. autodoc2-docstring:: pso.City.distance

   .. py:method:: tot_dist(new_vec)
      :canonical: pso.City.tot_dist

      .. autodoc2-docstring:: pso.City.tot_dist

   .. py:method:: _get_maxfes(ml)
      :canonical: pso.City._get_maxfes

      .. autodoc2-docstring:: pso.City._get_maxfes

.. py:class:: Particle(city)
   :canonical: pso.Particle

   .. autodoc2-docstring:: pso.Particle

   .. rubric:: Initialization

   .. autodoc2-docstring:: pso.Particle.__init__

   .. py:method:: evaluate(city_dist_matrix)
      :canonical: pso.Particle.evaluate

      .. autodoc2-docstring:: pso.Particle.evaluate

   .. py:method:: update_velocity(global_best_position, current_best_position, w, c1, c2)
      :canonical: pso.Particle.update_velocity

      .. autodoc2-docstring:: pso.Particle.update_velocity

   .. py:method:: calculate_velocity(position1, position2)
      :canonical: pso.Particle.calculate_velocity

      .. autodoc2-docstring:: pso.Particle.calculate_velocity

   .. py:method:: update_position()
      :canonical: pso.Particle.update_position

      .. autodoc2-docstring:: pso.Particle.update_position

.. py:class:: PSO(city, num_particles, max_iterations, w, c1, c2)
   :canonical: pso.PSO

   .. autodoc2-docstring:: pso.PSO

   .. rubric:: Initialization

   .. autodoc2-docstring:: pso.PSO.__init__

   .. py:method:: solve()
      :canonical: pso.PSO.solve

      .. autodoc2-docstring:: pso.PSO.solve

   .. py:method:: update_pso_best_solutions(fitness, position)
      :canonical: pso.PSO.update_pso_best_solutions

      .. autodoc2-docstring:: pso.PSO.update_pso_best_solutions

.. py:function:: update_best_solutions(fitness, position, best_solutions)
   :canonical: pso.update_best_solutions

   .. autodoc2-docstring:: pso.update_best_solutions
