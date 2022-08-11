#include "TestHelpers.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

using namespace Pennylane;
using Pennylane::Util::randomUnitary;

template <typename PrecisionT, class GateImplementation>
void testApplySingleQubitOp() {
    using ComplexPrecisionT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", Matrix0 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.203377341216, 0.132238554262},
            ComplexPrecisionT{0.216290940442, 0.203109511967},
            ComplexPrecisionT{0.290374372568, 0.123095338906},
            ComplexPrecisionT{0.040762810130, 0.153237600777},
            ComplexPrecisionT{0.062445212079, 0.106020046388},
            ComplexPrecisionT{0.041489260594, 0.149813636657},
            ComplexPrecisionT{0.002100244854, 0.099744848045},
            ComplexPrecisionT{0.281559630427, 0.083376695381},
            ComplexPrecisionT{0.073652349575, 0.066811372960},
            ComplexPrecisionT{0.150797357980, 0.146266222503},
            ComplexPrecisionT{0.324043781913, 0.157417591307},
            ComplexPrecisionT{0.040556496061, 0.254572386140},
            ComplexPrecisionT{0.204954964152, 0.098550445557},
            ComplexPrecisionT{0.056681743348, 0.225803880189},
            ComplexPrecisionT{0.327486634260, 0.130699704247},
            ComplexPrecisionT{0.299805387808, 0.150417378569},
        };

        const std::vector<size_t> wires = {0};
        std::vector<ComplexPrecisionT> matrix{
            ComplexPrecisionT{-0.6709485262524046, -0.6304426335363695},
            ComplexPrecisionT{-0.14885403153998722, 0.3608498832392019},
            ComplexPrecisionT{-0.2376311670004963, 0.3096798175687841},
            ComplexPrecisionT{-0.8818365947322423, -0.26456390390903695},
        };

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{-0.088159230256, -0.200310710166},
            ComplexPrecisionT{-0.092298136103, -0.239992165702},
            ComplexPrecisionT{-0.222261050476, -0.172156102614},
            ComplexPrecisionT{-0.028641644551, -0.151772474905},
            ComplexPrecisionT{-0.041128255234, -0.051213774079},
            ComplexPrecisionT{-0.023306864430, -0.139832054875},
            ComplexPrecisionT{-0.034436430308, 0.030470593143},
            ComplexPrecisionT{-0.235253129822, -0.147654159822},
            ComplexPrecisionT{-0.136553865781, -0.046844610803},
            ComplexPrecisionT{-0.208578251022, -0.150162656683},
            ComplexPrecisionT{-0.351228795803, -0.163875086963},
            ComplexPrecisionT{-0.025554644471, -0.259011641330},
            ComplexPrecisionT{-0.202335094297, -0.146984720225},
            ComplexPrecisionT{-0.046497880559, -0.236870070744},
            ComplexPrecisionT{-0.285599324363, -0.224949005764},
            ComplexPrecisionT{-0.317311776750, -0.144580799156},
        };

        auto st = ini_st;
        GateImplementation::applySingleQubitOp(st.data(), num_qubits,
                                               matrix.data(), wires, false);
        REQUIRE(st == approx(expected).margin(1e-5));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", Matrix1 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.203377341216, 0.132238554262},
            ComplexPrecisionT{0.216290940442, 0.203109511967},
            ComplexPrecisionT{0.290374372568, 0.123095338906},
            ComplexPrecisionT{0.040762810130, 0.153237600777},
            ComplexPrecisionT{0.062445212079, 0.106020046388},
            ComplexPrecisionT{0.041489260594, 0.149813636657},
            ComplexPrecisionT{0.002100244854, 0.099744848045},
            ComplexPrecisionT{0.281559630427, 0.083376695381},
            ComplexPrecisionT{0.073652349575, 0.066811372960},
            ComplexPrecisionT{0.150797357980, 0.146266222503},
            ComplexPrecisionT{0.324043781913, 0.157417591307},
            ComplexPrecisionT{0.040556496061, 0.254572386140},
            ComplexPrecisionT{0.204954964152, 0.098550445557},
            ComplexPrecisionT{0.056681743348, 0.225803880189},
            ComplexPrecisionT{0.327486634260, 0.130699704247},
            ComplexPrecisionT{0.299805387808, 0.150417378569},
        };

        const std::vector<size_t> wires = {1};
        std::vector<ComplexPrecisionT> matrix{
            ComplexPrecisionT{-0.06456334151703813, -0.46701592144475385},
            ComplexPrecisionT{-0.7849162862155974, -0.40203747049594296},
            ComplexPrecisionT{0.35001199887831924, 0.8094561783632672},
            ComplexPrecisionT{-0.4618134467695434, -0.09487168351817299},
        };

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.042236556846, -0.211840395533},
            ComplexPrecisionT{0.108556009215, -0.248396142329},
            ComplexPrecisionT{0.077192593353, -0.222692634399},
            ComplexPrecisionT{-0.118547567659, -0.207571660592},
            ComplexPrecisionT{-0.054636543376, 0.156024360645},
            ComplexPrecisionT{-0.093651051458, 0.173046696421},
            ComplexPrecisionT{0.010487272302, 0.231867409698},
            ComplexPrecisionT{-0.231889585995, 0.021414192236},
            ComplexPrecisionT{-0.094804784349, -0.198463810478},
            ComplexPrecisionT{0.104863870690, -0.279893530935},
            ComplexPrecisionT{-0.151908242161, -0.395747235633},
            ComplexPrecisionT{-0.058577814605, -0.273974623210},
            ComplexPrecisionT{-0.113602984277, 0.018046788174},
            ComplexPrecisionT{-0.070369209805, 0.063602025384},
            ComplexPrecisionT{-0.152841460397, 0.225969197890},
            ComplexPrecisionT{-0.316053740116, 0.024024286121},
        };

        auto st = ini_st;
        GateImplementation::applySingleQubitOp(st.data(), num_qubits,
                                               matrix.data(), wires, false);
        REQUIRE(st == approx(expected).margin(1e-5));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", Matrix2 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.203377341216, 0.132238554262},
            ComplexPrecisionT{0.216290940442, 0.203109511967},
            ComplexPrecisionT{0.290374372568, 0.123095338906},
            ComplexPrecisionT{0.040762810130, 0.153237600777},
            ComplexPrecisionT{0.062445212079, 0.106020046388},
            ComplexPrecisionT{0.041489260594, 0.149813636657},
            ComplexPrecisionT{0.002100244854, 0.099744848045},
            ComplexPrecisionT{0.281559630427, 0.083376695381},
            ComplexPrecisionT{0.073652349575, 0.066811372960},
            ComplexPrecisionT{0.150797357980, 0.146266222503},
            ComplexPrecisionT{0.324043781913, 0.157417591307},
            ComplexPrecisionT{0.040556496061, 0.254572386140},
            ComplexPrecisionT{0.204954964152, 0.098550445557},
            ComplexPrecisionT{0.056681743348, 0.225803880189},
            ComplexPrecisionT{0.327486634260, 0.130699704247},
            ComplexPrecisionT{0.299805387808, 0.150417378569},
        };

        const std::vector<size_t> wires = {2};
        std::vector<ComplexPrecisionT> matrix{
            ComplexPrecisionT{-0.09868517256862797, 0.1346373537372914},
            ComplexPrecisionT{-0.6272437275794093, 0.7607228969250281},
            ComplexPrecisionT{0.5083047843569023, 0.8448433380773042},
            ComplexPrecisionT{-0.07776920594546943, -0.14770893985433542},
        };

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{-0.313651523675, 0.158015857153},
            ComplexPrecisionT{-0.190830404547, -0.056031178288},
            ComplexPrecisionT{-0.012743088406, 0.186575564154},
            ComplexPrecisionT{-0.042189526065, 0.268035298797},
            ComplexPrecisionT{-0.097632230188, -0.063021774411},
            ComplexPrecisionT{-0.264297959806, 0.152692968180},
            ComplexPrecisionT{-0.043259258448, 0.098579615666},
            ComplexPrecisionT{-0.115061048825, 0.063129899779},
            ComplexPrecisionT{-0.339269297035, 0.151091393323},
            ComplexPrecisionT{-0.253672211262, -0.122958027429},
            ComplexPrecisionT{-0.020955943769, 0.036078832827},
            ComplexPrecisionT{-0.012472454359, 0.175959514614},
            ComplexPrecisionT{-0.338334782053, 0.185015137716},
            ComplexPrecisionT{-0.338472277486, 0.119068450947},
            ComplexPrecisionT{0.014757040712, 0.164711383267},
            ComplexPrecisionT{-0.163054937984, 0.106682609797},
        };

        auto st = ini_st;
        GateImplementation::applySingleQubitOp(st.data(), num_qubits,
                                               matrix.data(), wires, false);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}

template <typename PrecisionT, class GateImplementation>
void testApplyTwoQubitOp() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    DYNAMIC_SECTION(GateImplementation::name
                    << ", Matrix0,1 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.203377341216, 0.132238554262},
            ComplexPrecisionT{0.216290940442, 0.203109511967},
            ComplexPrecisionT{0.290374372568, 0.123095338906},
            ComplexPrecisionT{0.040762810130, 0.153237600777},
            ComplexPrecisionT{0.062445212079, 0.106020046388},
            ComplexPrecisionT{0.041489260594, 0.149813636657},
            ComplexPrecisionT{0.002100244854, 0.099744848045},
            ComplexPrecisionT{0.281559630427, 0.083376695381},
            ComplexPrecisionT{0.073652349575, 0.066811372960},
            ComplexPrecisionT{0.150797357980, 0.146266222503},
            ComplexPrecisionT{0.324043781913, 0.157417591307},
            ComplexPrecisionT{0.040556496061, 0.254572386140},
            ComplexPrecisionT{0.204954964152, 0.098550445557},
            ComplexPrecisionT{0.056681743348, 0.225803880189},
            ComplexPrecisionT{0.327486634260, 0.130699704247},
            ComplexPrecisionT{0.299805387808, 0.150417378569},
        };

        const std::vector<size_t> wires = {0, 1};
        std::vector<ComplexPrecisionT> matrix{
            ComplexPrecisionT{-0.010948839478141403, -0.4261536209511877},
            ComplexPrecisionT{-0.6522252885639435, -0.2941022724640708},
            ComplexPrecisionT{0.26225131765405274, 0.4236139177304751},
            ComplexPrecisionT{-0.12618692657772357, 0.20550327298620602},
            ComplexPrecisionT{-0.3628656010611667, 0.12279634811005145},
            ComplexPrecisionT{-0.299785993479099, 0.08052108649543024},
            ComplexPrecisionT{-0.3268760347664654, 0.29288358376151247},
            ComplexPrecisionT{0.06355368774014421, -0.7484828109139796},
            ComplexPrecisionT{-0.42696246581802144, 0.5312298019756412},
            ComplexPrecisionT{0.19082021416756584, -0.3848180554033438},
            ComplexPrecisionT{0.2905316308379382, 0.3567658763647916},
            ComplexPrecisionT{0.3051329918715622, 0.21495115396751993},
            ComplexPrecisionT{-0.04349597374383496, 0.45291155567640123},
            ComplexPrecisionT{-0.4540435198052925, 0.03313504573634055},
            ComplexPrecisionT{-0.5735449843600084, -0.1360234950731129},
            ComplexPrecisionT{-0.14650781658884487, 0.46562323100514574},
        };

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{-0.010522294886, -0.097227414150},
            ComplexPrecisionT{0.025218372075, -0.118918153811},
            ComplexPrecisionT{0.027356493011, 0.038593767440},
            ComplexPrecisionT{-0.260209778025, -0.029664187740},
            ComplexPrecisionT{-0.074148286001, -0.197175492308},
            ComplexPrecisionT{-0.047445065528, -0.120432480588},
            ComplexPrecisionT{-0.162531426347, -0.232103748235},
            ComplexPrecisionT{-0.080908114074, -0.339097688508},
            ComplexPrecisionT{-0.065452006637, 0.167593688919},
            ComplexPrecisionT{-0.174290686562, 0.218180044463},
            ComplexPrecisionT{-0.040769815236, 0.391540527230},
            ComplexPrecisionT{-0.032888113119, 0.062559768276},
            ComplexPrecisionT{-0.209674188196, 0.072947117389},
            ComplexPrecisionT{-0.305237717975, -0.088612881006},
            ComplexPrecisionT{-0.345917435397, 0.079914065740},
            ComplexPrecisionT{-0.304373922289, -0.050696747855},
        };

        auto st = ini_st;
        GateImplementation::applyTwoQubitOp(st.data(), num_qubits,
                                            matrix.data(), wires, false);
        REQUIRE(st == approx(expected).margin(1e-5));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", Matrix1,3 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.203377341216, 0.132238554262},
            ComplexPrecisionT{0.216290940442, 0.203109511967},
            ComplexPrecisionT{0.290374372568, 0.123095338906},
            ComplexPrecisionT{0.040762810130, 0.153237600777},
            ComplexPrecisionT{0.062445212079, 0.106020046388},
            ComplexPrecisionT{0.041489260594, 0.149813636657},
            ComplexPrecisionT{0.002100244854, 0.099744848045},
            ComplexPrecisionT{0.281559630427, 0.083376695381},
            ComplexPrecisionT{0.073652349575, 0.066811372960},
            ComplexPrecisionT{0.150797357980, 0.146266222503},
            ComplexPrecisionT{0.324043781913, 0.157417591307},
            ComplexPrecisionT{0.040556496061, 0.254572386140},
            ComplexPrecisionT{0.204954964152, 0.098550445557},
            ComplexPrecisionT{0.056681743348, 0.225803880189},
            ComplexPrecisionT{0.327486634260, 0.130699704247},
            ComplexPrecisionT{0.299805387808, 0.150417378569},
        };

        const std::vector<size_t> wires = {1, 3};
        std::vector<ComplexPrecisionT> matrix{
            ComplexPrecisionT{-0.4945444988183558, -0.11776474515763265},
            ComplexPrecisionT{-0.29362382883961335, 0.4309563356559181},
            ComplexPrecisionT{0.38642000389978287, -0.11761080702891993},
            ComplexPrecisionT{-0.4625432352960688, 0.3041708755288573},
            ComplexPrecisionT{-0.01670164516238861, -0.06290137757928804},
            ComplexPrecisionT{-0.5303475445655139, -0.281210965038839},
            ComplexPrecisionT{0.3210907020450156, -0.4908226494303132},
            ComplexPrecisionT{0.5371130944965893, 0.054034789251905496},
            ComplexPrecisionT{-0.6374608763985821, 0.1283980517316574},
            ComplexPrecisionT{0.11623580942989631, -0.5772052807682595},
            ComplexPrecisionT{-0.2801100593795251, -0.25593219011143187},
            ComplexPrecisionT{-0.24178695440574766, -0.16750226628447634},
            ComplexPrecisionT{0.43526551626023624, 0.35358616624711525},
            ComplexPrecisionT{0.12383614509048962, -0.07550807002819623},
            ComplexPrecisionT{0.3807245050274032, -0.45158286780318524},
            ComplexPrecisionT{-0.47191404042553947, -0.3047996016248278},
        };

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{-0.264205950272, -0.078825796270},
            ComplexPrecisionT{0.033605397647, -0.097442075257},
            ComplexPrecisionT{-0.350166020754, -0.037125900500},
            ComplexPrecisionT{0.220722295888, -0.022059951504},
            ComplexPrecisionT{0.020457336820, -0.248270874284},
            ComplexPrecisionT{0.181621494746, 0.067111057534},
            ComplexPrecisionT{-0.136891895494, -0.142700100623},
            ComplexPrecisionT{0.037867646910, 0.084010926977},
            ComplexPrecisionT{-0.139979818310, -0.092901195560},
            ComplexPrecisionT{0.096552234651, -0.070334396489},
            ComplexPrecisionT{-0.305840219133, -0.139674837753},
            ComplexPrecisionT{0.376774144027, -0.191209037401},
            ComplexPrecisionT{0.038354787323, -0.247322773715},
            ComplexPrecisionT{0.202764286721, -0.117020408763},
            ComplexPrecisionT{-0.180698521324, -0.259571676988},
            ComplexPrecisionT{0.197697750266, -0.048932641006},
        };

        auto st = ini_st;
        GateImplementation::applyTwoQubitOp(st.data(), num_qubits,
                                            matrix.data(), wires, false);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}

template <typename PrecisionT, class GateImplementation>
void testApplyMultiQubitOp() {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    DYNAMIC_SECTION(GateImplementation::name
                    << ", Matrix1,2,3 - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.203377341216, 0.132238554262},
            ComplexPrecisionT{0.216290940442, 0.203109511967},
            ComplexPrecisionT{0.290374372568, 0.123095338906},
            ComplexPrecisionT{0.040762810130, 0.153237600777},
            ComplexPrecisionT{0.062445212079, 0.106020046388},
            ComplexPrecisionT{0.041489260594, 0.149813636657},
            ComplexPrecisionT{0.002100244854, 0.099744848045},
            ComplexPrecisionT{0.281559630427, 0.083376695381},
            ComplexPrecisionT{0.073652349575, 0.066811372960},
            ComplexPrecisionT{0.150797357980, 0.146266222503},
            ComplexPrecisionT{0.324043781913, 0.157417591307},
            ComplexPrecisionT{0.040556496061, 0.254572386140},
            ComplexPrecisionT{0.204954964152, 0.098550445557},
            ComplexPrecisionT{0.056681743348, 0.225803880189},
            ComplexPrecisionT{0.327486634260, 0.130699704247},
            ComplexPrecisionT{0.299805387808, 0.150417378569},
        };

        const std::vector<size_t> wires = {1, 2, 3};
        std::vector<ComplexPrecisionT> matrix{
            ComplexPrecisionT{-0.14601911598243822, -0.18655250647340088},
            ComplexPrecisionT{-0.03917826201290317, -0.031161687050443518},
            ComplexPrecisionT{0.11497626236175404, 0.38310733543366354},
            ComplexPrecisionT{-0.0929691815340695, 0.1219804125497268},
            ComplexPrecisionT{0.07306514883467692, 0.017445444816725875},
            ComplexPrecisionT{-0.27330866098918355, -0.6007032759764033},
            ComplexPrecisionT{0.4530754397715841, -0.08267189625512258},
            ComplexPrecisionT{0.32125201986075, -0.036845158875036116},
            ComplexPrecisionT{0.032317572838307884, 0.02292755555300329},
            ComplexPrecisionT{-0.18775945295623664, -0.060215004737844156},
            ComplexPrecisionT{-0.3093351335745536, -0.2061961962889725},
            ComplexPrecisionT{0.4216087567144761, 0.010534488410902099},
            ComplexPrecisionT{0.2769943541718527, -0.26016137877135465},
            ComplexPrecisionT{0.18727884147867532, 0.02830415812286322},
            ComplexPrecisionT{0.3367562196770689, -0.5250999173939218},
            ComplexPrecisionT{0.05770014289220745, 0.26595514845958573},
            ComplexPrecisionT{0.37885720163317027, 0.3110931426403546},
            ComplexPrecisionT{0.13436510737129648, -0.4083415934958021},
            ComplexPrecisionT{-0.5443665467635203, 0.2458343977310266},
            ComplexPrecisionT{-0.050346912365833024, 0.08709833123617361},
            ComplexPrecisionT{0.11505259829552131, 0.010155858056939438},
            ComplexPrecisionT{-0.2930849061531229, 0.019339259194141145},
            ComplexPrecisionT{0.011825409829453282, 0.011597907736881019},
            ComplexPrecisionT{-0.10565527258356637, -0.3113689446440079},
            ComplexPrecisionT{0.0273191284561944, -0.2479498526173881},
            ComplexPrecisionT{-0.5528072425836249, -0.06114469689935285},
            ComplexPrecisionT{-0.20560364740746587, -0.3800208994544297},
            ComplexPrecisionT{-0.008236143958221483, 0.3017421511504845},
            ComplexPrecisionT{0.04817188123334976, 0.08550951191632741},
            ComplexPrecisionT{-0.24081054643565586, -0.3412671345149831},
            ComplexPrecisionT{-0.38913538197001885, 0.09288402897806938},
            ComplexPrecisionT{-0.07937578245883717, 0.013979426755633685},
            ComplexPrecisionT{0.22246583652015395, -0.18276674810033927},
            ComplexPrecisionT{0.22376666162382491, 0.2995723155125488},
            ComplexPrecisionT{-0.1727191441070097, -0.03880522034607489},
            ComplexPrecisionT{0.075780203819001, 0.2818783673816625},
            ComplexPrecisionT{-0.6161322400651016, 0.26067347179217193},
            ComplexPrecisionT{-0.021161519614267765, -0.08430919051054794},
            ComplexPrecisionT{0.1676500381348944, -0.30645601624407504},
            ComplexPrecisionT{-0.28858251997285883, 0.018089595494883842},
            ComplexPrecisionT{-0.19590767481842053, -0.12844366632033652},
            ComplexPrecisionT{0.18707834504831794, -0.1363932722670649},
            ComplexPrecisionT{-0.07224221779769334, -0.11267803536286894},
            ComplexPrecisionT{-0.23897684826459387, -0.39609971967853685},
            ComplexPrecisionT{-0.0032110880452929555, -0.29294331305690136},
            ComplexPrecisionT{-0.3188741682462722, -0.17338979346647143},
            ComplexPrecisionT{0.08194395032821632, -0.002944814673179825},
            ComplexPrecisionT{-0.5695791830944521, 0.33299548924055095},
            ComplexPrecisionT{-0.4983660307441444, -0.4222358493977972},
            ComplexPrecisionT{0.05533914327048402, -0.42575842134560576},
            ComplexPrecisionT{-0.2187623521182678, -0.03087596187054778},
            ComplexPrecisionT{0.11278255885846857, 0.07075886163492914},
            ComplexPrecisionT{-0.3054684775292515, -0.1739796870866232},
            ComplexPrecisionT{0.14151567663565712, 0.20399935744127418},
            ComplexPrecisionT{0.06720165377364941, 0.07543463072363207},
            ComplexPrecisionT{0.08019665306716581, -0.3473013434358584},
            ComplexPrecisionT{-0.2600167605995786, -0.08795704036197827},
            ComplexPrecisionT{0.125680477777759, 0.266342700305046},
            ComplexPrecisionT{-0.1586772594600269, 0.187360909108502},
            ComplexPrecisionT{-0.4653314704208982, 0.4048609954619629},
            ComplexPrecisionT{0.39992560380733094, -0.10029244177901954},
            ComplexPrecisionT{0.2533527906886461, 0.05222114898540775},
            ComplexPrecisionT{-0.15840033949128557, -0.2727320427534386},
            ComplexPrecisionT{-0.21590866323269536, -0.1191163626522938},
        };

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.140662010561, 0.048572894309},
            ComplexPrecisionT{0.021854656046, 0.079393113826},
            ComplexPrecisionT{-0.069686446335, -0.072998481807},
            ComplexPrecisionT{-0.126819464879, -0.378029601439},
            ComplexPrecisionT{-0.134716141061, 0.034463497732},
            ComplexPrecisionT{-0.054908086720, -0.157073288683},
            ComplexPrecisionT{0.005790250878, -0.346174821950},
            ComplexPrecisionT{-0.204004872847, 0.019278209615},
            ComplexPrecisionT{0.336927171579, 0.074028686268},
            ComplexPrecisionT{0.170822794112, -0.062115684096},
            ComplexPrecisionT{-0.133087934403, -0.164577625932},
            ComplexPrecisionT{-0.240412977475, -0.331519081061},
            ComplexPrecisionT{-0.228436573919, -0.063017646940},
            ComplexPrecisionT{-0.016556534913, -0.258822480482},
            ComplexPrecisionT{-0.012416037504, -0.214182329161},
            ComplexPrecisionT{-0.204751961090, -0.130791666115},
        };

        auto st = ini_st;
        GateImplementation::applyMultiQubitOp(st.data(), num_qubits,
                                              matrix.data(), wires, false);
        REQUIRE(st == approx(expected).margin(1e-5));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", Matrix0,1,2,3 - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexPrecisionT> ini_st{
            ComplexPrecisionT{0.203377341216, 0.132238554262},
            ComplexPrecisionT{0.216290940442, 0.203109511967},
            ComplexPrecisionT{0.290374372568, 0.123095338906},
            ComplexPrecisionT{0.040762810130, 0.153237600777},
            ComplexPrecisionT{0.062445212079, 0.106020046388},
            ComplexPrecisionT{0.041489260594, 0.149813636657},
            ComplexPrecisionT{0.002100244854, 0.099744848045},
            ComplexPrecisionT{0.281559630427, 0.083376695381},
            ComplexPrecisionT{0.073652349575, 0.066811372960},
            ComplexPrecisionT{0.150797357980, 0.146266222503},
            ComplexPrecisionT{0.324043781913, 0.157417591307},
            ComplexPrecisionT{0.040556496061, 0.254572386140},
            ComplexPrecisionT{0.204954964152, 0.098550445557},
            ComplexPrecisionT{0.056681743348, 0.225803880189},
            ComplexPrecisionT{0.327486634260, 0.130699704247},
            ComplexPrecisionT{0.299805387808, 0.150417378569},
        };

        const std::vector<size_t> wires = {0, 1, 2, 3};
        std::vector<ComplexPrecisionT> matrix{
            ComplexPrecisionT{-0.0811773464755885, -0.19682208345860647},
            ComplexPrecisionT{-0.06700740455243999, -0.04561583597315822},
            ComplexPrecisionT{0.46656064487776394, -0.20461647097832134},
            ComplexPrecisionT{0.010256952837273701, 0.139986239661604},
            ComplexPrecisionT{0.09467881261623481, -0.03336335480963736},
            ComplexPrecisionT{0.15936811587631455, 0.057315624600284346},
            ComplexPrecisionT{0.17978576553142844, 0.15678425145109015},
            ComplexPrecisionT{0.18055277474308914, -0.1165523390317797},
            ComplexPrecisionT{0.012281901962826983, 0.18007645868598343},
            ComplexPrecisionT{0.03284274273649668, 0.10947017370431167},
            ComplexPrecisionT{0.3432582684584454, -0.06867050921150765},
            ComplexPrecisionT{-0.10561876215199961, 0.07949429539442246},
            ComplexPrecisionT{0.19701386373457638, 0.16981005094001442},
            ComplexPrecisionT{0.13534591755792408, 0.28066932476781636},
            ComplexPrecisionT{0.1837983164589069, -0.23569832312461225},
            ComplexPrecisionT{-0.27499236539110833, -0.10791973216625947},
            ComplexPrecisionT{0.1834816172494385, -0.034827779112586055},
            ComplexPrecisionT{-0.0009230068403018204, -0.029512495868369862},
            ComplexPrecisionT{-0.05901102512885115, -0.15390444461487718},
            ComplexPrecisionT{0.05688461261840959, -0.03889302125105607},
            ComplexPrecisionT{0.29577734654421267, -0.13100455290792246},
            ComplexPrecisionT{0.3391919314190099, -0.2243138830097492},
            ComplexPrecisionT{0.16273296736459222, -0.2670701510726166},
            ComplexPrecisionT{-0.03336090672169414, 0.2455179411098639},
            ComplexPrecisionT{-0.3787491481884647, -0.11947116040140429},
            ComplexPrecisionT{-0.24173307291148638, -0.2213075797523435},
            ComplexPrecisionT{0.036982779372098394, -0.03544917593630506},
            ComplexPrecisionT{0.08741852442523015, 0.03010491205939035},
            ComplexPrecisionT{-0.010647043925551675, 0.37808566195485366},
            ComplexPrecisionT{-0.010545077863258843, -0.24546296648810367},
            ComplexPrecisionT{0.1316231994514309, -0.06760672518348157},
            ComplexPrecisionT{0.016131973166901414, -0.03503746492310948},
            ComplexPrecisionT{-0.06492487598324831, -0.1748331563535617},
            ComplexPrecisionT{0.031460705076800655, 0.11334377057699434},
            ComplexPrecisionT{0.03441902155646885, 0.22408157859399114},
            ComplexPrecisionT{0.0371280750051708, -0.1804040960738852},
            ComplexPrecisionT{0.09535644451753002, -0.025614187155079796},
            ComplexPrecisionT{-0.09216255279186078, 0.05633363422606912},
            ComplexPrecisionT{-0.11634815903744167, -0.23538903164817157},
            ComplexPrecisionT{-0.3039552254656059, -0.20552815005438066},
            ComplexPrecisionT{-0.27859216491701827, 0.10853204476017964},
            ComplexPrecisionT{0.24626461042473888, -0.1518650171214598},
            ComplexPrecisionT{0.4642911826057222, -0.11812007476730525},
            ComplexPrecisionT{0.36313614431900554, -0.14249281948146095},
            ComplexPrecisionT{-0.06380575042422873, -0.23288959902354697},
            ComplexPrecisionT{0.044361592585855265, -0.019957902264200526},
            ComplexPrecisionT{0.030605075870618798, -0.058098104432514286},
            ComplexPrecisionT{-0.014272674386900611, -0.14520609081384306},
            ComplexPrecisionT{0.23277420359739226, -0.14947000483002895},
            ComplexPrecisionT{-0.16342330755739196, 0.22676118594557015},
            ComplexPrecisionT{-0.20398531466518455, -0.28009454020028807},
            ComplexPrecisionT{-0.07735297055263991, 0.10081191135850101},
            ComplexPrecisionT{0.022873612354914773, -0.03126243586780419},
            ComplexPrecisionT{0.08881705356321942, -0.004514840514670637},
            ComplexPrecisionT{-0.05757185774631672, -0.13015459704290877},
            ComplexPrecisionT{-0.32070853617502854, -0.07803053060891867},
            ComplexPrecisionT{-0.07706926458785131, 0.05207299820519886},
            ComplexPrecisionT{0.2503902572322073, 0.09099135737552513},
            ComplexPrecisionT{-0.16383113911767921, 0.2973647186513588},
            ComplexPrecisionT{-0.27819268180170403, -0.30236394856248},
            ComplexPrecisionT{-0.12373346518281966, 0.03229258305240953},
            ComplexPrecisionT{-0.14477119991828383, 0.1959246121959782},
            ComplexPrecisionT{-0.045491424991441154, -0.032367020566301676},
            ComplexPrecisionT{-0.34719732324226155, -0.08638402027658323},
            ComplexPrecisionT{-0.0831137914503245, -0.38306179315772587},
            ComplexPrecisionT{0.33896013112876805, -0.08708321371958307},
            ComplexPrecisionT{0.08295290236493533, -0.12849960294752283},
            ComplexPrecisionT{0.3062034236838223, -0.16407337320867357},
            ComplexPrecisionT{-0.2525913492428881, 0.01270242036804025},
            ComplexPrecisionT{0.07387523797228966, 0.13236037063441145},
            ComplexPrecisionT{-0.10627680904019554, -0.1304826951782252},
            ComplexPrecisionT{-0.079498759063908, -0.10575816891195548},
            ComplexPrecisionT{0.12636058586101778, -0.3562639492682855},
            ComplexPrecisionT{-0.21351837585905967, -0.05490031570511319},
            ComplexPrecisionT{-0.08211886411050803, -0.014650442132510554},
            ComplexPrecisionT{0.1075526548380377, -0.09958704526495876},
            ComplexPrecisionT{0.1547843699970367, 0.18454687141005055},
            ComplexPrecisionT{-0.06315418425117045, 0.19385000729237473},
            ComplexPrecisionT{-0.18942009592159434, 0.23211726033840208},
            ComplexPrecisionT{-0.08358478431285932, -0.17542289710060982},
            ComplexPrecisionT{0.07414365811858471, 0.002553326538433997},
            ComplexPrecisionT{-0.12301246391656774, 0.08015352317475764},
            ComplexPrecisionT{-0.01987269429433799, 0.1569301408474106},
            ComplexPrecisionT{-0.12706461080431816, -0.12379705115137626},
            ComplexPrecisionT{-0.2758787728864479, 0.0035494406457515885},
            ComplexPrecisionT{-0.00984562112886961, -0.02481667008233526},
            ComplexPrecisionT{-0.084916806161764, 0.002014985096033997},
            ComplexPrecisionT{-0.04810808029083434, -0.0559974655611716},
            ComplexPrecisionT{-0.19559161234187358, -0.356244470773283},
            ComplexPrecisionT{-0.051130798099170774, 0.3205688860172856},
            ComplexPrecisionT{0.18301730588702717, -0.35462514629054254},
            ComplexPrecisionT{-0.29345027995280915, 0.3010185851398186},
            ComplexPrecisionT{-0.2572348744363236, 0.12734173180635364},
            ComplexPrecisionT{-0.07183902584877759, -0.08131174943117606},
            ComplexPrecisionT{-0.2170702641465722, -0.2294241732024488},
            ComplexPrecisionT{-0.1333742246142454, -0.09338397402993287},
            ComplexPrecisionT{0.2793032171967457, 0.24834849095463785},
            ComplexPrecisionT{-0.08087034521910429, 0.017228399318073954},
            ComplexPrecisionT{0.2736672517309202, -0.13557890045311372},
            ComplexPrecisionT{0.016893918658642156, -0.15032601746949797},
            ComplexPrecisionT{-0.4906452426501637, 0.16391653370174594},
            ComplexPrecisionT{0.05457620687567622, 0.060399646900382575},
            ComplexPrecisionT{-0.04607801770469559, -0.11126264743946929},
            ComplexPrecisionT{-0.046762700213043976, 0.2605200559933763},
            ComplexPrecisionT{0.03720080407993604, 0.12764613070765587},
            ComplexPrecisionT{0.06752680234074637, -0.003965034003632684},
            ComplexPrecisionT{0.28187184016252287, 0.26497003409635334},
            ComplexPrecisionT{0.05337188694517632, -0.08820558091508003},
            ComplexPrecisionT{0.22213111054177023, 0.051232875330921135},
            ComplexPrecisionT{0.08329146371445284, -0.3063899155813947},
            ComplexPrecisionT{-0.1661041551303006, 0.04669729713417685},
            ComplexPrecisionT{-0.07012401396611247, 0.09229606742901898},
            ComplexPrecisionT{-0.03268230910789402, -0.07891702877461022},
            ComplexPrecisionT{-0.6088891510900916, 0.010869529571772913},
            ComplexPrecisionT{0.181327487214985, -0.10980293958533298},
            ComplexPrecisionT{-0.06910826525269803, -0.002123158429041397},
            ComplexPrecisionT{0.17226502926121215, -0.0163737379577749},
            ComplexPrecisionT{-0.29034764218037307, 0.0010825500957120766},
            ComplexPrecisionT{-0.39606935617644395, -0.11929012808517281},
            ComplexPrecisionT{0.031140622944890596, -0.08978846977992205},
            ComplexPrecisionT{0.09829720632173958, -0.15419259877086008},
            ComplexPrecisionT{-0.16979247786848609, -0.20104559788220375},
            ComplexPrecisionT{0.022771727354803907, 0.07376310045777855},
            ComplexPrecisionT{0.10248457555947008, 0.0959848862914514},
            ComplexPrecisionT{-0.08500603537774072, 0.19979163374724185},
            ComplexPrecisionT{0.16183991510732185, 0.08834832692616706},
            ComplexPrecisionT{-0.07405592956498913, 0.17113318238915956},
            ComplexPrecisionT{-0.010865837528670281, 0.18262669089522315},
            ComplexPrecisionT{0.3007340238878014, 0.01228269465380516},
            ComplexPrecisionT{0.14841870857595918, 0.04052914054163879},
            ComplexPrecisionT{0.07979188546811528, 0.03448057959133652},
            ComplexPrecisionT{-0.08878394458437894, -0.2555136336296365},
            ComplexPrecisionT{0.05550210679913066, -0.13359457037788244},
            ComplexPrecisionT{-0.24170895206097853, -0.5657605420017089},
            ComplexPrecisionT{0.1366326110342102, 0.39911030119908275},
            ComplexPrecisionT{-0.057513002498134125, -0.056872501379940044},
            ComplexPrecisionT{0.12028326294930897, -0.17837416458259414},
            ComplexPrecisionT{0.12424194113702836, 0.024058926963841928},
            ComplexPrecisionT{0.053139541530483975, -0.05945051420622338},
            ComplexPrecisionT{0.11364580087339703, -0.21140604493359308},
            ComplexPrecisionT{-0.023179998082468196, 0.12855784626080657},
            ComplexPrecisionT{0.18447197799514206, 0.059283465491411},
            ComplexPrecisionT{0.02268886826432022, 0.14991864893901055},
            ComplexPrecisionT{-0.11684281491031034, 0.05965071504188227},
            ComplexPrecisionT{0.1644084622807226, 0.2504871714298593},
            ComplexPrecisionT{0.13677900982607485, 0.045751657613712624},
            ComplexPrecisionT{0.07637224308536966, 0.043949816587819344},
            ComplexPrecisionT{-0.15232748233545576, 0.39490236818664404},
            ComplexPrecisionT{-0.07620965894681739, -0.00044015958182986453},
            ComplexPrecisionT{0.015228807982348392, -0.03989508573959658},
            ComplexPrecisionT{-0.06025106319147798, -0.10646056582772014},
            ComplexPrecisionT{0.08404620644271446, 0.10988557291824569},
            ComplexPrecisionT{-0.24019669759696022, -0.32677139387437215},
            ComplexPrecisionT{-0.1818352969905923, -0.104919596974696},
            ComplexPrecisionT{0.05869841366621599, -0.041398191449316835},
            ComplexPrecisionT{0.145282212607238, -0.1924313363943447},
            ComplexPrecisionT{0.11060376483178569, -0.26643308349138073},
            ComplexPrecisionT{0.033457421541339696, 0.45989190483425835},
            ComplexPrecisionT{-0.11299501544002627, -0.1701839390369943},
            ComplexPrecisionT{-0.027076760012979297, 0.22500799348689948},
            ComplexPrecisionT{-0.12738473942173661, -0.2324834516745204},
            ComplexPrecisionT{-0.19309725929725693, 0.05397935052707938},
            ComplexPrecisionT{-0.1580199221099269, 0.02650400948671369},
            ComplexPrecisionT{-0.3243602236537279, -0.1222811623098372},
            ComplexPrecisionT{-0.3667586083634349, 0.009310446080170115},
            ComplexPrecisionT{0.319516408449643, -0.06325831841303045},
            ComplexPrecisionT{0.08903851358406503, 0.035815693787163774},
            ComplexPrecisionT{0.26343384552070426, -0.06458983496775506},
            ComplexPrecisionT{-0.015982429855605745, 0.021963169842707644},
            ComplexPrecisionT{0.1857171521950028, -0.18246521253046694},
            ComplexPrecisionT{-0.059754779213136586, 0.017676478816029573},
            ComplexPrecisionT{0.09776213545785536, -0.21762015291541528},
            ComplexPrecisionT{-0.1021998970995581, 0.17743735401036495},
            ComplexPrecisionT{0.204067980539901, 0.16726678221372981},
            ComplexPrecisionT{-0.0644818679085489, -0.1447270248131951},
            ComplexPrecisionT{0.39083699741955674, -0.03858837033831454},
            ComplexPrecisionT{0.0846216954671219, -0.12110810943095418},
            ComplexPrecisionT{-0.08992012094040822, -0.17734766719453254},
            ComplexPrecisionT{-0.04125741492613475, -0.1839167873488144},
            ComplexPrecisionT{-0.12059699651731241, 0.14869580875265737},
            ComplexPrecisionT{-0.2937373017282166, -0.22981405150756443},
            ComplexPrecisionT{0.08463640092878551, -0.11016426157266608},
            ComplexPrecisionT{0.01308530686553121, 0.3559206665166033},
            ComplexPrecisionT{-0.1331583159634414, -0.10956805576318897},
            ComplexPrecisionT{-0.1644759222332063, 0.09866893124404766},
            ComplexPrecisionT{-0.1900211227067346, -0.3658017080024861},
            ComplexPrecisionT{0.017273241356383206, 0.09372965786434638},
            ComplexPrecisionT{0.0773103659755795, 0.3489293973682007},
            ComplexPrecisionT{-0.08102734189288471, -0.33217362924578586},
            ComplexPrecisionT{-0.11464128747877839, -0.05875281079321294},
            ComplexPrecisionT{-0.014051241439616557, 0.1544645442809333},
            ComplexPrecisionT{-0.1123948350954748, -0.19432517027831833},
            ComplexPrecisionT{0.0330691662270549, -0.04919564002605478},
            ComplexPrecisionT{-0.1503779515321153, 0.31786458703670245},
            ComplexPrecisionT{0.20056990324454096, 0.1850109516828582},
            ComplexPrecisionT{0.2668695146420536, -0.43908623087339876},
            ComplexPrecisionT{0.011925951393575142, 0.17761991713542086},
            ComplexPrecisionT{0.005498164983530092, -0.002242630083480047},
            ComplexPrecisionT{0.17886013938198958, 0.10872382704642089},
            ComplexPrecisionT{0.08132470444007044, -0.09749221481303638},
            ComplexPrecisionT{-0.1737463651573049, 0.033299982055501054},
            ComplexPrecisionT{-0.18488118424023303, -0.3108837738476596},
            ComplexPrecisionT{-0.27113768160711277, 0.05256567395241565},
            ComplexPrecisionT{-0.19544234063361635, -0.01048324904429301},
            ComplexPrecisionT{0.0945949165859533, -0.2172498114363532},
            ComplexPrecisionT{-0.10985875532895437, 0.06386245713299038},
            ComplexPrecisionT{-0.025338698108748416, -0.23532608763027604},
            ComplexPrecisionT{-0.0916460338613978, 0.17111255669108807},
            ComplexPrecisionT{0.006942232587543371, 0.0449604620457603},
            ComplexPrecisionT{0.3550406307789137, -0.02112983773334885},
            ComplexPrecisionT{-0.03139393148937276, -0.12538378880991538},
            ComplexPrecisionT{-0.050516685188229764, 0.01574122670219121},
            ComplexPrecisionT{-0.1481888835160628, 0.13072587703855404},
            ComplexPrecisionT{-0.07986143820660621, 0.016012176121866912},
            ComplexPrecisionT{-0.32768187389890996, -0.015327242028981312},
            ComplexPrecisionT{0.2558846540681018, -0.3436385199742097},
            ComplexPrecisionT{-0.05851615752958035, -0.005701707401036289},
            ComplexPrecisionT{0.019780986059743003, -0.12316275013093546},
            ComplexPrecisionT{-0.09991905974897695, 0.14752267178535067},
            ComplexPrecisionT{0.1291642581984909, 0.023163549022428606},
            ComplexPrecisionT{-0.25784163164652146, 0.15076651333162944},
            ComplexPrecisionT{0.03232346409394317, -0.22680289177803026},
            ComplexPrecisionT{0.3358286037483417, -0.291011759563723},
            ComplexPrecisionT{-0.2598827120355048, 0.20093990553213004},
            ComplexPrecisionT{0.0182311108231829, -0.07832433364549145},
            ComplexPrecisionT{0.025583973820756032, -0.023411273206879912},
            ComplexPrecisionT{-0.1654037078192205, -0.16284081930646588},
            ComplexPrecisionT{0.0019996979156362543, -0.10309761662934443},
            ComplexPrecisionT{0.16916605119355627, 0.18306162041718788},
            ComplexPrecisionT{-0.2559127189443478, 0.09222694309162274},
            ComplexPrecisionT{-0.09919028489941913, 0.18613169533463675},
            ComplexPrecisionT{0.07508213832551346, 0.41694487884583353},
            ComplexPrecisionT{0.023972824357252787, -0.06599789836959391},
            ComplexPrecisionT{0.016773306946815114, -0.03939241313465466},
            ComplexPrecisionT{-0.08664317188243345, 0.12563189408611994},
            ComplexPrecisionT{0.21026595312801138, 0.08176642742039242},
            ComplexPrecisionT{0.0093028207667191, -0.06279692109628966},
            ComplexPrecisionT{0.2203850153337698, 0.020898065538940646},
            ComplexPrecisionT{-0.20583392564574995, -0.43652451614186233},
            ComplexPrecisionT{-0.08855804365999222, -0.4392951412878797},
            ComplexPrecisionT{0.07533134195667648, 0.4606082840651167},
            ComplexPrecisionT{0.052524607577778604, -0.023110444690372434},
            ComplexPrecisionT{-0.30000883176081444, -0.20888279683044061},
            ComplexPrecisionT{0.07247961742346525, -0.2320002192036349},
            ComplexPrecisionT{-0.053018678053434604, 0.01691219103432881},
            ComplexPrecisionT{-0.2399217664380185, 0.16673537527457852},
            ComplexPrecisionT{0.0916908309563268, -0.04065242875621084},
            ComplexPrecisionT{0.13288223871943822, -0.1449122646529056},
            ComplexPrecisionT{-0.1440177353834986, 0.20921513414099863},
            ComplexPrecisionT{-0.019495585563170055, -0.22503492052552065},
            ComplexPrecisionT{0.17343212478867365, -0.1494555146156185},
            ComplexPrecisionT{-0.12018617624189008, 0.056351189991110794},
            ComplexPrecisionT{-0.0016827409660573889, 0.25808153364327724},
            ComplexPrecisionT{-0.04741488633872766, 0.36934530803669297},
            ComplexPrecisionT{-0.002345825594843698, 0.16427813741863367},
            ComplexPrecisionT{0.023944547993878525, -0.17484976236216634},
        };

        std::vector<ComplexPrecisionT> expected{
            ComplexPrecisionT{0.264749262466, -0.016918881870},
            ComplexPrecisionT{0.201841741500, 0.074598631106},
            ComplexPrecisionT{0.260163664155, -0.025732021493},
            ComplexPrecisionT{-0.321148080521, -0.224394633392},
            ComplexPrecisionT{0.057117880510, -0.098204010637},
            ComplexPrecisionT{-0.199622088528, -0.357183553512},
            ComplexPrecisionT{0.142361283082, 0.304993936113},
            ComplexPrecisionT{-0.167235905884, -0.079450883404},
            ComplexPrecisionT{0.223735534585, 0.156113785239},
            ComplexPrecisionT{-0.039724326748, 0.074784370874},
            ComplexPrecisionT{0.128903821272, -0.052607394218},
            ComplexPrecisionT{-0.100102973432, -0.369701144889},
            ComplexPrecisionT{-0.076618826943, -0.113689447069},
            ComplexPrecisionT{0.137136222122, -0.081190249787},
            ComplexPrecisionT{-0.054059628740, -0.174640023638},
            ComplexPrecisionT{-0.073454475362, -0.053685843736},
        };

        auto st = ini_st;
        GateImplementation::applyMultiQubitOp(st.data(), num_qubits,
                                              matrix.data(), wires, false);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}

template <typename PrecisionT, typename TypeList>
void testApplyMatrixForKernels() {
    using Gates::MatrixOperation;
    if constexpr (!std::is_same_v<TypeList, void>) {
        using GateImplementation = typename TypeList::Type;

        if constexpr (Util::array_has_elt(
                          GateImplementation::implemented_matrices,
                          MatrixOperation::SingleQubitOp)) {
            testApplySingleQubitOp<PrecisionT, GateImplementation>();
        }
        if constexpr (Util::array_has_elt(
                          GateImplementation::implemented_matrices,
                          MatrixOperation::TwoQubitOp)) {
            testApplyTwoQubitOp<PrecisionT, GateImplementation>();
        }
        if constexpr (Util::array_has_elt(
                          GateImplementation::implemented_matrices,
                          MatrixOperation::MultiQubitOp)) {
            testApplyMultiQubitOp<PrecisionT, GateImplementation>();
        }
        testApplyMatrixForKernels<PrecisionT, typename TypeList::Next>();
    }
}

TEMPLATE_TEST_CASE("GateImplementation::applyMatrix, inverse = false",
                   "[GateImplementations_Matrix]", float, double) {
    using PrecisionT = TestType;

    testApplyMatrixForKernels<PrecisionT, TestKernels>();
}

template <typename PrecisionT, class GateImplementation>
void testApplySingleQubitOpInverse() {
    std::mt19937 re{1337};
    const int num_qubits = 4;
    const auto margin = PrecisionT{1e-5};

    DYNAMIC_SECTION(GateImplementation::name
                    << ", wires = {0} - "
                    << PrecisionToName<PrecisionT>::value) {
        const std::vector<size_t> wires{0};

        const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);

        const auto matrix = randomUnitary<PrecisionT>(re, wires.size());

        auto st = ini_st;
        GateImplementation::applySingleQubitOp(st.data(), num_qubits,
                                               matrix.data(), wires, false);
        GateImplementation::applySingleQubitOp(st.data(), num_qubits,
                                               matrix.data(), wires, true);
        REQUIRE(st == approx(ini_st).margin(margin));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", wires = {1} - "
                    << PrecisionToName<PrecisionT>::value) {
        const std::vector<size_t> wires{1};

        const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);

        const auto matrix = randomUnitary<PrecisionT>(re, wires.size());

        auto st = ini_st;
        GateImplementation::applySingleQubitOp(st.data(), num_qubits,
                                               matrix.data(), wires, false);
        GateImplementation::applySingleQubitOp(st.data(), num_qubits,
                                               matrix.data(), wires, true);

        REQUIRE(st == approx(ini_st).margin(margin));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", wires = {2} - "
                    << PrecisionToName<PrecisionT>::value) {
        const std::vector<size_t> wires{2};

        const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);

        const auto matrix = randomUnitary<PrecisionT>(re, wires.size());

        auto st = ini_st;
        GateImplementation::applySingleQubitOp(st.data(), num_qubits,
                                               matrix.data(), wires, false);
        GateImplementation::applySingleQubitOp(st.data(), num_qubits,
                                               matrix.data(), wires, true);

        REQUIRE(st == approx(ini_st).margin(margin));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", wires = {3} - "
                    << PrecisionToName<PrecisionT>::value) {
        const std::vector<size_t> wires{3};

        const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);

        const auto matrix = randomUnitary<PrecisionT>(re, wires.size());

        auto st = ini_st;
        GateImplementation::applySingleQubitOp(st.data(), num_qubits,
                                               matrix.data(), wires, false);
        GateImplementation::applySingleQubitOp(st.data(), num_qubits,
                                               matrix.data(), wires, true);

        REQUIRE(st == approx(ini_st).margin(margin));
    }
}

template <typename PrecisionT, class GateImplementation>
void testApplyTwoQubitOpInverse() {
    std::mt19937 re{1337};
    const int num_qubits = 4;
    const auto margin = PrecisionT{1e-5};
    DYNAMIC_SECTION(GateImplementation::name
                    << ", wires = {0,1} - "
                    << PrecisionToName<PrecisionT>::value) {
        const std::vector<size_t> wires{0, 1};

        const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);

        const auto matrix = randomUnitary<PrecisionT>(re, wires.size());

        auto st = ini_st;
        GateImplementation::applyTwoQubitOp(st.data(), num_qubits,
                                            matrix.data(), wires, false);
        GateImplementation::applyTwoQubitOp(st.data(), num_qubits,
                                            matrix.data(), wires, true);

        REQUIRE(st == approx(ini_st).margin(margin));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", wires = {1,2} - "
                    << PrecisionToName<PrecisionT>::value) {
        const std::vector<size_t> wires{1, 2};

        const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);
        const auto matrix = randomUnitary<PrecisionT>(re, wires.size());

        auto st = ini_st;
        GateImplementation::applyTwoQubitOp(st.data(), num_qubits,
                                            matrix.data(), wires, false);
        GateImplementation::applyTwoQubitOp(st.data(), num_qubits,
                                            matrix.data(), wires, true);

        REQUIRE(st == approx(ini_st).margin(margin));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", wires = {1,3} - "
                    << PrecisionToName<PrecisionT>::value) {
        const std::vector<size_t> wires{1, 3};

        const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);
        const auto matrix = randomUnitary<PrecisionT>(re, wires.size());

        auto st = ini_st;
        GateImplementation::applyTwoQubitOp(st.data(), num_qubits,
                                            matrix.data(), wires, false);
        GateImplementation::applyTwoQubitOp(st.data(), num_qubits,
                                            matrix.data(), wires, true);

        REQUIRE(st == approx(ini_st).margin(margin));
    }
}

template <typename PrecisionT, class GateImplementation>
void testApplyMultiQubitOpInverse() {
    std::mt19937 re{1337};
    const int num_qubits = 4;
    const auto margin = PrecisionT{1e-5};

    DYNAMIC_SECTION(GateImplementation::name
                    << ", wires = {1,2,3} - "
                    << PrecisionToName<PrecisionT>::value) {
        const std::vector<size_t> wires{1, 2, 3};

        const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);
        const auto matrix = randomUnitary<PrecisionT>(re, wires.size());

        auto st = ini_st;
        GateImplementation::applyMultiQubitOp(st.data(), num_qubits,
                                              matrix.data(), wires, false);
        GateImplementation::applyMultiQubitOp(st.data(), num_qubits,
                                              matrix.data(), wires, true);

        REQUIRE(st == approx(ini_st).margin(margin));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", wires = {0,1,2,3} - "
                    << PrecisionToName<PrecisionT>::value) {
        const std::vector<size_t> wires{0, 1, 2, 3};
        const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);

        const auto matrix = randomUnitary<PrecisionT>(re, wires.size());

        auto st = ini_st;
        GateImplementation::applyMultiQubitOp(st.data(), num_qubits,
                                              matrix.data(), wires, false);
        GateImplementation::applyMultiQubitOp(st.data(), num_qubits,
                                              matrix.data(), wires, true);

        REQUIRE(st == approx(ini_st).margin(margin));
    }
}

template <typename PrecisionT, typename TypeList>
void testApplyMatrixInverseForKernels() {
    using Gates::MatrixOperation;
    if constexpr (!std::is_same_v<TypeList, void>) {
        using GateImplementation = typename TypeList::Type;
        if constexpr (Util::array_has_elt(
                          GateImplementation::implemented_matrices,
                          MatrixOperation::SingleQubitOp)) {
            testApplySingleQubitOpInverse<PrecisionT, GateImplementation>();
        }
        if constexpr (Util::array_has_elt(
                          GateImplementation::implemented_matrices,
                          MatrixOperation::TwoQubitOp)) {
            testApplyTwoQubitOpInverse<PrecisionT, GateImplementation>();
        }
        if constexpr (Util::array_has_elt(
                          GateImplementation::implemented_matrices,
                          MatrixOperation::MultiQubitOp)) {
            testApplyMultiQubitOpInverse<PrecisionT, GateImplementation>();
        }
        testApplyMatrixInverseForKernels<PrecisionT, typename TypeList::Next>();
    }
}

TEMPLATE_TEST_CASE("GateImplementation::applyMatrix, inverse = true",
                   "[GateImplementations_Matrix]", float, double) {
    using PrecisionT = TestType;

    testApplyMatrixInverseForKernels<PrecisionT, TestKernels>();
}
