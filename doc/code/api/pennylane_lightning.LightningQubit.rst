LightningQubit
=============================

.. currentmodule:: pennylane_lightning

.. autoclass:: LightningQubit
   :show-inheritance:

   .. raw:: html

      <a class="attr-details-header collapse-header" data-toggle="collapse" href="#attrDetails" aria-expanded="false" aria-controls="attrDetails">
         <h2 style="font-size: 24px;">
            <i class="fas fa-angle-down rotate" style="float: right;"></i> Attributes
         </h2>
      </a>
      <div class="collapse" id="attrDetails">

   .. autosummary::
      :nosignatures:

      ~LightningQubit.analytic
      ~LightningQubit.author
      ~LightningQubit.circuit_hash
      ~LightningQubit.measurement_map
      ~LightningQubit.name
      ~LightningQubit.num_executions
      ~LightningQubit.obs_queue
      ~LightningQubit.observables
      ~LightningQubit.op_queue
      ~LightningQubit.operations
      ~LightningQubit.parameters
      ~LightningQubit.pennylane_requires
      ~LightningQubit.short_name
      ~LightningQubit.shot_vector
      ~LightningQubit.shots
      ~LightningQubit.state
      ~LightningQubit.stopping_condition
      ~LightningQubit.version
      ~LightningQubit.wire_map
      ~LightningQubit.wires

   .. autoattribute:: analytic
   .. autoattribute:: author
   .. autoattribute:: circuit_hash
   .. autoattribute:: measurement_map
   .. autoattribute:: name
   .. autoattribute:: num_executions
   .. autoattribute:: obs_queue
   .. autoattribute:: observables
   .. autoattribute:: op_queue
   .. autoattribute:: operations
   .. autoattribute:: parameters
   .. autoattribute:: pennylane_requires
   .. autoattribute:: short_name
   .. autoattribute:: shot_vector
   .. autoattribute:: shots
   .. autoattribute:: state
   .. autoattribute:: stopping_condition
   .. autoattribute:: version
   .. autoattribute:: wire_map
   .. autoattribute:: wires

   .. raw:: html

      </div>

   .. raw:: html

      <a class="meth-details-header collapse-header" data-toggle="collapse" href="#methDetails" aria-expanded="false" aria-controls="methDetails">
         <h2 style="font-size: 24px;">
            <i class="fas fa-angle-down rotate" style="float: right;"></i> Methods
         </h2>
      </a>
      <div class="collapse" id="methDetails">

   .. autosummary::

      ~LightningQubit.access_state
      ~LightningQubit.active_wires
      ~LightningQubit.adjoint_jacobian
      ~LightningQubit.analytic_probability
      ~LightningQubit.apply
      ~LightningQubit.apply_lightning
      ~LightningQubit.batch_execute
      ~LightningQubit.batch_transform
      ~LightningQubit.batch_vjp
      ~LightningQubit.capabilities
      ~LightningQubit.check_validity
      ~LightningQubit.classical_shadow
      ~LightningQubit.custom_expand
      ~LightningQubit.default_expand_fn
      ~LightningQubit.define_wire_map
      ~LightningQubit.density_matrix
      ~LightningQubit.estimate_probability
      ~LightningQubit.execute
      ~LightningQubit.execute_and_gradients
      ~LightningQubit.execution_context
      ~LightningQubit.expand_fn
      ~LightningQubit.expval
      ~LightningQubit.generate_basis_states
      ~LightningQubit.generate_samples
      ~LightningQubit.gradients
      ~LightningQubit.map_wires
      ~LightningQubit.marginal_prob
      ~LightningQubit.mutual_info
      ~LightningQubit.order_wires
      ~LightningQubit.post_apply
      ~LightningQubit.post_measure
      ~LightningQubit.pre_apply
      ~LightningQubit.pre_measure
      ~LightningQubit.probability
      ~LightningQubit.reset
      ~LightningQubit.sample
      ~LightningQubit.sample_basis_states
      ~LightningQubit.shadow_expval
      ~LightningQubit.shot_vec_statistics
      ~LightningQubit.states_to_binary
      ~LightningQubit.statistics
      ~LightningQubit.supports_observable
      ~LightningQubit.supports_operation
      ~LightningQubit.var
      ~LightningQubit.vjp
      ~LightningQubit.vn_entropy

   .. automethod:: access_state
   .. automethod:: active_wires
   .. automethod:: adjoint_jacobian
   .. automethod:: analytic_probability
   .. automethod:: apply
   .. automethod:: apply_lightning
   .. automethod:: batch_execute
   .. automethod:: batch_transform
   .. automethod:: batch_vjp
   .. automethod:: capabilities
   .. automethod:: check_validity
   .. automethod:: classical_shadow
   .. automethod:: custom_expand
   .. automethod:: default_expand_fn
   .. automethod:: define_wire_map
   .. automethod:: density_matrix
   .. automethod:: estimate_probability
   .. automethod:: execute
   .. automethod:: execute_and_gradients
   .. automethod:: execution_context
   .. automethod:: expand_fn
   .. automethod:: expval
   .. automethod:: generate_basis_states
   .. automethod:: generate_samples
   .. automethod:: gradients
   .. automethod:: map_wires
   .. automethod:: marginal_prob
   .. automethod:: mutual_info
   .. automethod:: order_wires
   .. automethod:: post_apply
   .. automethod:: post_measure
   .. automethod:: pre_apply
   .. automethod:: pre_measure
   .. automethod:: probability
   .. automethod:: reset
   .. automethod:: sample
   .. automethod:: sample_basis_states
   .. automethod:: shadow_expval
   .. automethod:: shot_vec_statistics
   .. automethod:: states_to_binary
   .. automethod:: statistics
   .. automethod:: supports_observable
   .. automethod:: supports_operation
   .. automethod:: var
   .. automethod:: vjp
   .. automethod:: vn_entropy

   .. raw:: html

      </div>

   .. raw:: html

      <script type="text/javascript">
         $(".collapse-header").click(function () {
             $(this).children('h2').eq(0).children('i').eq(0).toggleClass("up");
         })
      </script>
